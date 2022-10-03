use colored::Colorize;
use cust::prelude::*;
use nanorand::{Rng, WyRand};
use std::error::Error;
use std::time::Instant;

/// How many numbers to generate and add together.
const NUMBERS_LEN: usize = 100_000_000;

static PTX: &str = include_str!("../../../resources/add.ptx");

fn main() -> Result<(), Box<dyn Error>> {
    let _result_add = addition()?;
    let _result_mult = multipy()?;
    Ok(())
}

fn init_2_arrays() -> (Vec<f32>, Vec<f32>) {
    let mut wyrand = WyRand::new();
    let mut lhs = vec![2.0f32; NUMBERS_LEN];
    wyrand.fill(&mut lhs);
    let mut rhs = vec![0.0f32; NUMBERS_LEN];
    wyrand.fill(&mut rhs);
    (lhs, rhs)
}

fn addition() -> Result<Vec<f32>, Box<dyn Error>> {
    let (lhs, rhs) = init_2_arrays();

    let mut out_gpu = vec![0.0f32; NUMBERS_LEN];
    let mut out_cpu = vec![0.0f32; NUMBERS_LEN];
    println!(
        "{}",
        format!("Testing on adding {NUMBERS_LEN} size f32 slice").blue()
    );
    add_gpu(&lhs, &rhs, &mut out_gpu)?;

    add_cpu(&lhs, &rhs, &mut out_cpu);
    for i in 0..NUMBERS_LEN - 1 {
        assert_eq!(
            out_cpu[i], out_gpu[i],
            "Some calculations went wrong! {} != {}",
            out_cpu[i], out_gpu[i]
        )
    }
    Ok(out_gpu)
}

fn add_cpu(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
    let before = Instant::now();
    for i in 0..NUMBERS_LEN - 1 {
        out[i] = lhs[i] + rhs[i];
    }
    println!("Elapsed time using CPU: {:.2?}", before.elapsed());
}

fn add_gpu(lhs: &[f32], rhs: &[f32], out: &mut [f32]) -> Result<(), Box<dyn Error>> {
    // initialize CUDA, this will pick the first available device and will
    // make a CUDA context from it.
    // We don't need the context for anything but it must be kept alive.
    let _ctx = cust::quick_init()?;
    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // they can be made from PTX code, cubins, or fatbins.
    let module = Module::from_ptx(PTX, &[])?;
    // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
    // GPU calls.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    // allocate the GPU memory needed to house our numbers and copy them over.
    let lhs_gpu = lhs.as_dbuf()?;
    let rhs_gpu = rhs.as_dbuf()?;
    // allocate our output buffer. You could also use DeviceBuffer::uninitialized() to avoid the
    // cost of the copy, but you need to be careful not to read from the buffer.
    let out_buf = out.as_ref().as_dbuf()?;
    // retrieve the add kernel from the module so we can calculate the right launch config.
    let func = module.get_function("add")?;
    // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
    // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
    // current CUDA device/architecture.
    let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;
    let grid_size = (NUMBERS_LEN as u32 + block_size - 1) / block_size;
    let before = Instant::now();
    println!(
        "using {} blocks and {} threads per block",
        grid_size, block_size
    );
    // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
    // not block the thread until the kernel is finished.
    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            func<<<grid_size, block_size, 0, stream>>>(
                lhs_gpu.as_device_ptr(),
                lhs_gpu.len(),
                rhs_gpu.as_device_ptr(),
                rhs_gpu.len(),
                out_buf.as_device_ptr(),
            )
        )?;
    }
    stream.synchronize()?;
    // copy back the data from the GPU.
    out_buf.copy_to(out)?;
    println!("Elapsed time using GPU: {:.2?}", before.elapsed());
    Ok(())
}

fn multipy() -> Result<Vec<f32>, Box<dyn Error>> {
    let (lhs, rhs) = init_2_arrays();

    let mut out_gpu = vec![0.0f32; NUMBERS_LEN];
    let mut out_cpu = vec![0.0f32; NUMBERS_LEN];
    println!(
        "{}",
        format!("Testing on multipling {NUMBERS_LEN} size f32 slice").green()
    );
    multiple_gpu(&lhs, &rhs, &mut out_gpu)?;

    multiple_cpu(&lhs, &rhs, &mut out_cpu);
    for i in 0..NUMBERS_LEN - 1 {
        assert_eq!(
            out_cpu[i], out_gpu[i],
            "Some calculations went wrong! {} != {}",
            out_cpu[i], out_gpu[i]
        )
    }
    Ok(out_gpu)
}

fn multiple_cpu(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
    let before = Instant::now();
    for i in 0..NUMBERS_LEN - 1 {
        out[i] = lhs[i] * rhs[i];
    }
    println!("Elapsed time using CPU: {:.2?}", before.elapsed());
}

fn multiple_gpu(lhs: &[f32], rhs: &[f32], out: &mut [f32]) -> Result<(), Box<dyn Error>> {
    // initialize CUDA, this will pick the first available device and will
    // make a CUDA context from it.
    // We don't need the context for anything but it must be kept alive.
    let _ctx = cust::quick_init()?;
    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // they can be made from PTX code, cubins, or fatbins.
    let module = Module::from_ptx(PTX, &[])?;
    // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
    // GPU calls.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    // allocate the GPU memory needed to house our numbers and copy them over.
    let lhs_gpu = lhs.as_dbuf()?;
    let rhs_gpu = rhs.as_dbuf()?;
    // allocate our output buffer. You could also use DeviceBuffer::uninitialized() to avoid the
    // cost of the copy, but you need to be careful not to read from the buffer.
    let out_buf = out.as_ref().as_dbuf()?;
    // retrieve the add kernel from the module so we can calculate the right launch config.
    let func = module.get_function("multiple")?;
    // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
    // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
    // current CUDA device/architecture.
    let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;
    let grid_size = (NUMBERS_LEN as u32 + block_size - 1) / block_size;
    let before = Instant::now();
    println!(
        "using {} blocks and {} threads per block",
        grid_size, block_size
    );
    // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
    // not block the thread until the kernel is finished.
    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            func<<<grid_size, block_size, 0, stream>>>(
                lhs_gpu.as_device_ptr(),
                lhs_gpu.len(),
                rhs_gpu.as_device_ptr(),
                rhs_gpu.len(),
                out_buf.as_device_ptr(),
            )
        )?;
    }
    stream.synchronize()?;
    // copy back the data from the GPU.
    out_buf.copy_to(out)?;
    println!("Elapsed time using GPU: {:.2?}", before.elapsed());
    Ok(())
}
