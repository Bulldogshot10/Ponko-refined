use std::env;

fn main() {
    let cpu_flags = env::var("PONKO_CPU_ARCH_FLAGS").unwrap_or_else(|_| {
        "-march=armv8.6-a+dotprod+i8mm+fp16 -mtune=cortex-x4".to_owned()
    });
    let vulkan_enabled = env::var("PONKO_VULKAN_ENABLED").unwrap_or_else(|_| "OFF".to_owned());

    println!("cargo:rerun-if-env-changed=PONKO_CPU_ARCH_FLAGS");
    println!("cargo:rerun-if-env-changed=PONKO_VULKAN_ENABLED");
    println!("cargo:rustc-env=PONKO_CPU_ARCH_FLAGS={cpu_flags}");
    println!("cargo:rustc-env=PONKO_VULKAN_ENABLED={vulkan_enabled}");
}
