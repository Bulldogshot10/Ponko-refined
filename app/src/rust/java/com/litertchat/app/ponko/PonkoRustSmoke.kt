// This source set is compiled with the default Rust/UniFFI facade.
// Compiling it verifies that generated Kotlin bindings match the Rust exports.
package com.litertchat.app.ponko

object PonkoRustSmoke {
    fun runtimeDescription(): String {
        val version = engineApiVersion()
        val profile = buildProfile()
        check(version == profile.apiVersion) {
            "Ponko Rust facade API mismatch: function=$version profile=${profile.apiVersion}"
        }
        return "${profile.abi} | ${profile.cpuArchFlags} | API v$version"
    }
}
