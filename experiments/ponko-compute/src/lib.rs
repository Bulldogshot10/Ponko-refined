//! Minimal user-space Vulkan compute runtime prototype.
//!
//! This first slice keeps the path explicit: host-visible storage buffers,
//! one compute queue, SPIR-V pipelines, descriptor sets, dispatch, and a fence.
//! It is a measurement harness, not yet a tuned tensor library.

pub mod benchmark;

use anyhow::{bail, ensure, Context, Result};
use ash::{vk, Device, Entry, Instance};
use std::{
    ffi::{CStr, CString},
    fs,
    path::Path,
    ptr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

struct Inner {
    _entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    queue: vk::Queue,
    queue_mutex: Mutex<()>,
    queue_family_index: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    max_push_constant_size: u32,
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A Vulkan compute device using the system loader and currently installed driver.
pub struct ComputeDevice {
    inner: Arc<Inner>,
}

/// A host-visible Vulkan storage buffer.
pub struct ComputeBuffer {
    inner: Arc<Inner>,
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl ComputeBuffer {
    pub fn size_bytes(&self) -> u64 {
        self.size
    }
}

impl Drop for ComputeBuffer {
    fn drop(&mut self) {
        unsafe {
            self.inner.device.destroy_buffer(self.handle, None);
            self.inner.device.free_memory(self.memory, None);
        }
    }
}

/// A compiled SPIR-V compute pipeline and its binding layout.
pub struct ComputeKernel {
    inner: Arc<Inner>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    binding_count: u32,
    push_constant_bytes: u32,
}

impl Drop for ComputeKernel {
    fn drop(&mut self) {
        unsafe {
            self.inner.device.destroy_pipeline(self.pipeline, None);
            self.inner
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.inner
                .device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
        }
    }
}

impl ComputeDevice {
    /// Open the first Vulkan physical device that exposes a compute-capable queue.
    pub fn open() -> Result<Self> {
        let entry = unsafe { Entry::load() }.context("load the system Vulkan loader")?;
        let app_name = CString::new("PonkoCompute")?;
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&app_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_0);
        let instance_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .context("create Vulkan instance")?;

        let physical_devices = match unsafe { instance.enumerate_physical_devices() } {
            Ok(devices) => devices,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                return Err(error).context("enumerate Vulkan physical devices");
            }
        };
        if physical_devices.is_empty() {
            unsafe { instance.destroy_instance(None) };
            bail!("Vulkan loader exposed no physical devices");
        }

        let mut selected = None;
        for physical in physical_devices {
            let queues = unsafe { instance.get_physical_device_queue_family_properties(physical) };
            if let Some((index, _)) = queues.iter().enumerate().find(|(_, queue)| {
                queue.queue_count > 0 && queue.queue_flags.contains(vk::QueueFlags::COMPUTE)
            }) {
                selected = Some((physical, index as u32));
                break;
            }
        }
        let (physical_device, queue_family_index) = match selected {
            Some(selection) => selection,
            None => {
                unsafe { instance.destroy_instance(None) };
                bail!("no Vulkan compute queue found");
            }
        };

        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)
            .build()];
        let device_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info);
        let device = match unsafe { instance.create_device(physical_device, &device_info, None) } {
            Ok(device) => device,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                return Err(error).context("create Vulkan logical device");
            }
        };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let inner = Inner {
            _entry: entry,
            instance,
            physical_device,
            device,
            queue,
            queue_mutex: Mutex::new(()),
            queue_family_index,
            memory_properties,
            max_push_constant_size: properties.limits.max_push_constants_size,
        };
        println!("PonkoCompute selected Vulkan device: {device_name}");
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// The physical-device name, so benchmark output can be tied to the actual GPU.
    pub fn device_name(&self) -> String {
        let properties = unsafe {
            self.inner
                .instance
                .get_physical_device_properties(self.inner.physical_device)
        };
        unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    /// Allocate a host-visible/coherent storage buffer for the initial UMA-oriented probe.
    pub fn create_buffer(&self, size_bytes: usize) -> Result<ComputeBuffer> {
        ensure!(size_bytes > 0, "buffer size must be non-zero");
        let info = vk::BufferCreateInfo::builder()
            .size(size_bytes as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let handle = unsafe { self.inner.device.create_buffer(&info, None) }
            .context("create storage buffer")?;
        let requirements = unsafe { self.inner.device.get_buffer_memory_requirements(handle) };
        let required =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory_type = (0..self.inner.memory_properties.memory_type_count).find(|index| {
            (requirements.memory_type_bits & (1_u32 << *index)) != 0
                && self.inner.memory_properties.memory_types[*index as usize]
                    .property_flags
                    .contains(required)
        });
        let memory_type_index = match memory_type {
            Some(index) => index,
            None => {
                unsafe { self.inner.device.destroy_buffer(handle, None) };
                bail!("no HOST_VISIBLE | HOST_COHERENT memory type for storage buffers");
            }
        };
        let allocation = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = match unsafe { self.inner.device.allocate_memory(&allocation, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { self.inner.device.destroy_buffer(handle, None) };
                return Err(error).context("allocate storage-buffer memory");
            }
        };
        if let Err(error) = unsafe { self.inner.device.bind_buffer_memory(handle, memory, 0) } {
            unsafe {
                self.inner.device.free_memory(memory, None);
                self.inner.device.destroy_buffer(handle, None);
            }
            return Err(error).context("bind storage-buffer memory");
        }
        Ok(ComputeBuffer {
            inner: Arc::clone(&self.inner),
            handle,
            memory,
            size: size_bytes as u64,
        })
    }

    /// Copy raw bytes into a host-visible buffer.
    pub fn upload(&self, buffer: &ComputeBuffer, bytes: &[u8]) -> Result<()> {
        self.ensure_owned(buffer)?;
        ensure!(
            bytes.len() as u64 <= buffer.size,
            "upload exceeds buffer size"
        );
        if bytes.is_empty() {
            return Ok(());
        }
        let mapped = unsafe {
            self.inner.device.map_memory(
                buffer.memory,
                0,
                bytes.len() as u64,
                vk::MemoryMapFlags::empty(),
            )
        }
        .context("map buffer for upload")?;
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
            self.inner.device.unmap_memory(buffer.memory);
        }
        Ok(())
    }

    /// Read raw bytes from a host-visible buffer after a completed dispatch.
    pub fn readback(&self, buffer: &ComputeBuffer, byte_count: usize) -> Result<Vec<u8>> {
        self.ensure_owned(buffer)?;
        ensure!(
            byte_count as u64 <= buffer.size,
            "readback exceeds buffer size"
        );
        if byte_count == 0 {
            return Ok(Vec::new());
        }
        let mut bytes = vec![0_u8; byte_count];
        let mapped = unsafe {
            self.inner.device.map_memory(
                buffer.memory,
                0,
                byte_count as u64,
                vk::MemoryMapFlags::empty(),
            )
        }
        .context("map buffer for readback")?;
        unsafe {
            ptr::copy_nonoverlapping(mapped.cast::<u8>(), bytes.as_mut_ptr(), byte_count);
            self.inner.device.unmap_memory(buffer.memory);
        }
        Ok(bytes)
    }

    /// Load a SPIR-V compute shader. Each descriptor binding represents one storage buffer.
    /// The push-constant size must be zero or 4-byte aligned.
    pub fn load_kernel(
        &self,
        spirv_path: impl AsRef<Path>,
        binding_count: u32,
        push_constant_bytes: u32,
    ) -> Result<ComputeKernel> {
        ensure!(
            binding_count > 0,
            "kernel must declare at least one storage-buffer binding"
        );
        ensure!(
            push_constant_bytes % 4 == 0,
            "push-constant size must be 4-byte aligned"
        );
        ensure!(
            push_constant_bytes <= self.inner.max_push_constant_size,
            "push constants exceed this device's limit"
        );

        let bytes = fs::read(spirv_path.as_ref())
            .with_context(|| format!("read SPIR-V file {}", spirv_path.as_ref().display()))?;
        ensure!(
            !bytes.is_empty() && bytes.len() % 4 == 0,
            "SPIR-V size must be a non-zero multiple of four"
        );
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let shader_info = vk::ShaderModuleCreateInfo::builder().code(&words);
        let shader = unsafe { self.inner.device.create_shader_module(&shader_info, None) }
            .context("create SPIR-V shader module")?;

        let bindings: Vec<_> = (0..binding_count)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build()
            })
            .collect();
        let descriptor_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let descriptor_layout = match unsafe {
            self.inner
                .device
                .create_descriptor_set_layout(&descriptor_info, None)
        } {
            Ok(layout) => layout,
            Err(error) => {
                unsafe { self.inner.device.destroy_shader_module(shader, None) };
                return Err(error).context("create descriptor-set layout");
            }
        };

        let set_layouts = [descriptor_layout];
        let push_ranges = if push_constant_bytes > 0 {
            vec![vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push_constant_bytes)
                .build()]
        } else {
            Vec::new()
        };
        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        let pipeline_layout =
            match unsafe { self.inner.device.create_pipeline_layout(&layout_info, None) } {
                Ok(layout) => layout,
                Err(error) => {
                    unsafe {
                        self.inner
                            .device
                            .destroy_descriptor_set_layout(descriptor_layout, None);
                        self.inner.device.destroy_shader_module(shader, None);
                    }
                    return Err(error).context("create compute pipeline layout");
                }
            };

        let entry_name = CString::new("main")?;
        let stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(&entry_name);
        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage.build())
            .layout(pipeline_layout);
        let pipeline_result = unsafe {
            self.inner.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info.build()],
                None,
            )
        };
        unsafe { self.inner.device.destroy_shader_module(shader, None) };
        let pipeline = match pipeline_result {
            Ok(mut pipelines) => pipelines.remove(0),
            Err((_, error)) => {
                unsafe {
                    self.inner
                        .device
                        .destroy_pipeline_layout(pipeline_layout, None);
                    self.inner
                        .device
                        .destroy_descriptor_set_layout(descriptor_layout, None);
                }
                return Err(error).context("create compute pipeline");
            }
        };

        Ok(ComputeKernel {
            inner: Arc::clone(&self.inner),
            pipeline,
            pipeline_layout,
            descriptor_layout,
            binding_count,
            push_constant_bytes,
        })
    }

    /// Bind buffers in descriptor order, dispatch workgroups, and wait for completion.
    /// Returns wall time from queue submission through the completion fence.
    pub fn dispatch(
        &self,
        kernel: &ComputeKernel,
        buffers: &[&ComputeBuffer],
        push_constants: &[u8],
        workgroups: [u32; 3],
    ) -> Result<Duration> {
        ensure!(
            Arc::ptr_eq(&self.inner, &kernel.inner),
            "kernel belongs to another device"
        );
        ensure!(
            buffers.len() == kernel.binding_count as usize,
            "wrong number of storage buffers"
        );
        ensure!(
            push_constants.len() == kernel.push_constant_bytes as usize,
            "push-constant byte count does not match pipeline layout"
        );
        ensure!(
            workgroups.iter().all(|&count| count > 0),
            "workgroup counts must be non-zero"
        );
        for buffer in buffers {
            self.ensure_owned(buffer)?;
        }

        let pool_sizes = [vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(kernel.binding_count)
            .build()];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { self.inner.device.create_descriptor_pool(&pool_info, None) }
            .context("create descriptor pool")?;
        let set_layouts = [kernel.descriptor_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_sets =
            match unsafe { self.inner.device.allocate_descriptor_sets(&alloc_info) } {
                Ok(sets) => sets,
                Err(error) => {
                    unsafe {
                        self.inner
                            .device
                            .destroy_descriptor_pool(descriptor_pool, None)
                    };
                    return Err(error).context("allocate descriptor set");
                }
            };
        let descriptor_set = descriptor_sets[0];
        let buffer_infos: Vec<_> = buffers
            .iter()
            .map(|buffer| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(buffer.handle)
                    .offset(0)
                    .range(buffer.size)
                    .build()
            })
            .collect();
        let writes: Vec<_> = buffer_infos
            .iter()
            .enumerate()
            .map(|(index, info)| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(index as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
                    .build()
            })
            .collect();
        unsafe { self.inner.device.update_descriptor_sets(&writes, &[]) };

        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(self.inner.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let command_pool = match unsafe {
            self.inner
                .device
                .create_command_pool(&command_pool_info, None)
        } {
            Ok(pool) => pool,
            Err(error) => {
                unsafe {
                    self.inner
                        .device
                        .destroy_descriptor_pool(descriptor_pool, None)
                };
                return Err(error).context("create transient command pool");
            }
        };
        let alloc_commands = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers =
            match unsafe { self.inner.device.allocate_command_buffers(&alloc_commands) } {
                Ok(commands) => commands,
                Err(error) => {
                    unsafe {
                        self.inner.device.destroy_command_pool(command_pool, None);
                        self.inner
                            .device
                            .destroy_descriptor_pool(descriptor_pool, None);
                    }
                    return Err(error).context("allocate command buffer");
                }
            };
        let command = command_buffers[0];
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if let Err(error) = unsafe { self.inner.device.begin_command_buffer(command, &begin_info) }
        {
            unsafe {
                self.inner.device.destroy_command_pool(command_pool, None);
                self.inner
                    .device
                    .destroy_descriptor_pool(descriptor_pool, None);
            }
            return Err(error).context("begin command buffer");
        }
        unsafe {
            self.inner.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline,
            );
            self.inner.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            if !push_constants.is_empty() {
                self.inner.device.cmd_push_constants(
                    command,
                    kernel.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
            self.inner
                .device
                .cmd_dispatch(command, workgroups[0], workgroups[1], workgroups[2]);
            let barriers: Vec<_> = buffers
                .iter()
                .map(|buffer| {
                    vk::BufferMemoryBarrier::builder()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(buffer.handle)
                        .offset(0)
                        .size(buffer.size)
                        .build()
                })
                .collect();
            self.inner.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
        }
        if let Err(error) = unsafe { self.inner.device.end_command_buffer(command) } {
            unsafe {
                self.inner.device.destroy_command_pool(command_pool, None);
                self.inner
                    .device
                    .destroy_descriptor_pool(descriptor_pool, None);
            }
            return Err(error).context("end command buffer");
        }

        let fence = match unsafe {
            self.inner
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => fence,
            Err(error) => {
                unsafe {
                    self.inner.device.destroy_command_pool(command_pool, None);
                    self.inner
                        .device
                        .destroy_descriptor_pool(descriptor_pool, None);
                }
                return Err(error).context("create dispatch fence");
            }
        };
        let commands = [command];
        let submit = [vk::SubmitInfo::builder().command_buffers(&commands).build()];
        // VkQueue operations require external synchronization. Serialize submissions
        // when callers share a ComputeDevice across threads.
        let _queue_guard = self
            .inner
            .queue_mutex
            .lock()
            .map_err(|_| anyhow::anyhow!("Vulkan queue submission lock poisoned"))?;
        let start = Instant::now();
        let submit_result = unsafe {
            self.inner
                .device
                .queue_submit(self.inner.queue, &submit, fence)
        };
        if let Err(error) = submit_result {
            unsafe {
                self.inner.device.destroy_fence(fence, None);
                self.inner.device.destroy_command_pool(command_pool, None);
                self.inner
                    .device
                    .destroy_descriptor_pool(descriptor_pool, None);
            }
            return Err(error).context("submit compute work");
        }
        let wait_result = unsafe { self.inner.device.wait_for_fences(&[fence], true, u64::MAX) };
        let elapsed = start.elapsed();
        unsafe {
            self.inner.device.destroy_fence(fence, None);
            self.inner.device.destroy_command_pool(command_pool, None);
            self.inner
                .device
                .destroy_descriptor_pool(descriptor_pool, None);
        }
        wait_result.context("wait for compute fence")?;
        Ok(elapsed)
    }

    fn ensure_owned(&self, buffer: &ComputeBuffer) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.inner, &buffer.inner),
            "buffer belongs to another device"
        );
        Ok(())
    }
}
