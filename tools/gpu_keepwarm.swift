// Experimental: keeps the GPU lightly busy from a separate process with one 32-thread SIMD-group,
// to measure and reduce the SSD-streaming decode slowdown. See results/gpu-keepwarm.md.
// It uses power for as long as it runs, also while the server is idle. Stop it with Ctrl-C.
//
//   swiftc -O tools/gpu_keepwarm.swift -o gpu-keepwarm
//   ./gpu-keepwarm            # optional argument: loop iterations per buffer, default 20000
import Metal
import Foundation

let source = """
#include <metal_stdlib>
using namespace metal;
kernel void spin(device float * out [[buffer(0)]], constant uint & n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    float x = float(i) + 1.0f;
    for (uint k = 0; k < n; ++k) { x = fma(x, 0.9999f, 0.5f); }
    out[i] = x;
}
"""
let device = MTLCreateSystemDefaultDevice()!
let pipeline = try device.makeComputePipelineState(function: try device.makeLibrary(source: source, options: nil).makeFunction(name: "spin")!)
let queue = device.makeCommandQueue()!
let out = device.makeBuffer(length: 32 * 4, options: .storageModeShared)!
var iterations: UInt32 = UInt32(CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "20000")!
var inFlight: [MTLCommandBuffer] = []
var count = 0, busy = 0.0
let started = Date()
var running = true
let stops = [SIGINT, SIGTERM, SIGHUP].map { number -> DispatchSourceSignal in
    signal(number, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
    source.setEventHandler { running = false }
    source.resume()
    return source
}
while running {
    autoreleasepool {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(out, offset: 0, index: 0)
        enc.setBytes(&iterations, length: 4, index: 1)
        enc.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        inFlight.append(cb)
        if inFlight.count >= 2 {
            let first = inFlight.removeFirst()
            first.waitUntilCompleted()
            count += 1
            busy += first.gpuEndTime - first.gpuStartTime
        }
    }
}
_ = stops
print("keepwarm buffers=\(count) mean_ms=\(1000 * busy / Double(max(count, 1))) wall_s=\(Date().timeIntervalSince(started))")
