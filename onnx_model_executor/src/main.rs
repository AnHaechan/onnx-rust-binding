// Compact inference session, by reducing c_api_samples.rs
// reference code: https://github.com/nbigaouette/onnxruntime-rs/blob/master/onnxruntime-sys/examples/c_api_sample.rs

use std::os::unix::ffi::OsStrExt;
use std::{ffi::c_void, ffi::CString, os::raw::c_char, ptr::null};
use onnxruntime_sys::*;
use std::ptr::copy;
use std::time::Instant;

struct Config {
    device: Device,
    graph_opt_level: GraphOptimizationLevel,
    onnx_model_path: String,
    io_shape : IOShape,
    calib_table_and_engine_cache_path: CString,
    calib_table_generated_by: CalibTableGeneratedBy,
}

enum Device {
    Gpu { device_id : i32 },
    // parallel execution : only supported for Cpu, as Gpu is already parallel enough
    Cpu { enable_parallel_execution : bool }, 
}

struct IOShape {
    // FIXME: now only support single-input, single-output
    // but may have multiple inputs, such as `OnnxRuntime Resize`
    input_shape: Vec<u64>,
    output_shape: Vec<u64>,
    out_name: String,
}

enum CalibTableGeneratedBy {
    OnnxRuntime,
    TensorRT,
}

fn main() {
    // example configuration
    // let config = Config {
    //     device: Device::Gpu { device_id: 0 },
    //     graph_opt_level: GraphOptimizationLevel::ORT_ENABLE_EXTENDED,
    //     onnx_model_path:  "squeezenet1.0-12.onnx".to_string(),
    //     io_shape : IOShape { 
    //         input_shape: vec![1, 3, 224, 224], 
    //         output_shape: vec![1, 1000, 1, 1], 
    //         out_name: "softmaxout_1".to_string(), 
    //     },
    // };

    // let config2 = Config {
    //     device: Device::Cpu { enable_parallel_execution: true },
    //     graph_opt_level: GraphOptimizationLevel::ORT_ENABLE_EXTENDED,
    //     onnx_model_path: "squeezenet1.0-12.onnx".to_string(),
    //     io_shape : IOShape { 
    //         input_shape: vec![1, 3, 224, 224], 
    //         output_shape: vec![1, 1000, 1, 1],
    //         out_name: "softmaxout_1".to_string(), 
    //     },
    // };

    // let config3 = Config {
    //     device: Device::Gpu { device_id: 0 },
    //     graph_opt_level: GraphOptimizationLevel::ORT_ENABLE_EXTENDED,
    //     onnx_model_path: "CSPDarkNet53.onnx".to_string(),
    //     io_shape: IOShape { 
    //         input_shape: vec![1, 3, 256, 256], 
    //         output_shape: vec![1, 1000],
    //         out_name: "706_dequantized".to_string(),
    //     },
    // };
    
    let config4 = Config {
        device: Device::Gpu { device_id: 0 },
        graph_opt_level: GraphOptimizationLevel::ORT_ENABLE_EXTENDED,
        onnx_model_path: "CSPDarkNet53_f32.onnx".to_string(),
        io_shape: IOShape { 
            input_shape: vec![1, 3, 256, 256], 
            output_shape: vec![1, 1000],
            out_name: "771".to_string(),
        },
        calib_table_and_engine_cache_path: CString::new("CSPDarkNet53_f32_cache").unwrap(),
        calib_table_generated_by: CalibTableGeneratedBy::OnnxRuntime,
    };

    let _ = load_and_infer(config4);
}

fn load_and_infer(config: Config) -> Vec<f32> {
    // Interface for onnx runtime settings
    let g_ort = unsafe { OrtGetApiBase().as_ref().unwrap().GetApi.unwrap()(ORT_API_VERSION) };
    assert_ne!(g_ort, std::ptr::null_mut());

    //*************************************************************************
    // initialize  enviroment...one enviroment per process
    // enviroment maintains thread pools and other state info
    let mut env_ptr: *mut OrtEnv = std::ptr::null_mut();
    let env_name = std::ffi::CString::new("test").unwrap();
    let status = unsafe {
        g_ort.as_ref().unwrap().CreateEnv.unwrap()(
            OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
            env_name.as_ptr(),
            &mut env_ptr,
        )
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(env_ptr, std::ptr::null_mut());

    // initialize session options if needed
    let mut session_options_ptr: *mut OrtSessionOptions = std::ptr::null_mut();
    
    let status =
        unsafe { g_ort.as_ref().unwrap().CreateSessionOptions.unwrap()(&mut session_options_ptr) };
    CheckStatus(g_ort, status).unwrap();
    
    unsafe { g_ort.as_ref().unwrap().SetIntraOpNumThreads.unwrap()(session_options_ptr, 1) };
    assert_ne!(session_options_ptr, std::ptr::null_mut());
    
    if let Device::Cpu{ enable_parallel_execution } = &config.device {
        let execution_mode = if *enable_parallel_execution {
            ExecutionMode::ORT_PARALLEL  
        } else {
            ExecutionMode::ORT_SEQUENTIAL
        };
        unsafe { g_ort.as_ref().unwrap().SetSessionExecutionMode.unwrap()(session_options_ptr, execution_mode) };
    }

    // Sets graph optimization level
    unsafe {
        g_ort
            .as_ref()
            .unwrap()
            .SetSessionGraphOptimizationLevel
            .unwrap()(
            session_options_ptr,
            config.graph_opt_level,
        )
    };

    // // CUDA execution provider for Gpu
    // if let Device::Gpu { device_id } = config.device {
    //     let cuda_options: *const OrtCUDAProviderOptions = &OrtCUDAProviderOptions {
    //         device_id,
    //         cudnn_conv_algo_search: OrtCudnnConvAlgoSearch::EXHAUSTIVE,
    //         gpu_mem_limit: usize::MAX,
    //         arena_extend_strategy: 0,
    //         do_copy_in_default_stream: 1,
    //         has_user_compute_stream: 0,
    //         user_compute_stream: null::<c_void>() as *mut c_void,
    //         default_memory_arena_cfg: null::<OrtArenaCfg>() as *mut OrtArenaCfg,
    //     };
    
    //     unsafe {
    //         g_ort
    //             .as_ref()
    //             .unwrap()
    //             .SessionOptionsAppendExecutionProvider_CUDA
    //             .unwrap()(session_options_ptr, cuda_options);
    //     };
    // }

    // TensorRT execution provider for Gpu 
    // Use default value from https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#environment-variables
    if let Device::Gpu { device_id } = config.device {
        let trt_options: *const OrtTensorRTProviderOptions = &OrtTensorRTProviderOptions {
            device_id,
            has_user_compute_stream: 0,
            user_compute_stream: null::<c_void>() as *mut c_void,
            trt_max_workspace_size: 2147483648,
            trt_max_partition_iterations: 1000,
            trt_min_subgraph_size: 1,
            trt_fp16_enable: 0,
            trt_dla_enable: 0,
            trt_dla_core: 0,
            trt_force_sequential_engine_build: 0,
            // use or not? dump subgraphs transfomred into trt format to the filesystem
            trt_dump_subgraphs: 0, 
            // must use for i8 quantization
            trt_int8_enable: 1,
            trt_int8_calibration_table_name: CString::new("calibration.flatbuffers").unwrap().into_raw(),
            trt_int8_use_native_calibration_table: 
                match config.calib_table_generated_by {
                    CalibTableGeneratedBy::TensorRT => 1,
                    CalibTableGeneratedBy::OnnxRuntime => 0,
                },
            // must use for calibration table & engine caching
            trt_engine_cache_enable: 1,
            trt_engine_cache_path: config.calib_table_and_engine_cache_path.as_ptr(),
            trt_engine_decryption_enable: 0,
            trt_engine_decryption_lib_path: null::<c_char>(),
        };

        unsafe {
            g_ort
                .as_ref()
                .unwrap()
                .SessionOptionsAppendExecutionProvider_TensorRT
                .unwrap()(session_options_ptr, trt_options);
        }
    }

    //*************************************************************************
    // create session and load model into memory
    // NOTE: Original C version loaded SqueezeNet 1.0 (ONNX version: 1.3, Opset version: 8,
    //       https://github.com/onnx/models/blob/master/vision/classification/squeezenet/model/squeezenet1.0-8.onnx)
    //       Download it:
    //           curl -LO "https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.0-8.onnx"
    //       Reference: https://github.com/onnx/models/tree/master/vision/classification/squeezenet#model

    let model_path = std::ffi::OsString::from(config.onnx_model_path);
    let model_path: Vec<std::os::raw::c_char> = model_path
        .as_bytes()
        .iter()
        .chain(std::iter::once(&b'\0')) // Make sure we have a null terminated string
        .map(|b| *b as std::os::raw::c_char)
        .collect();

    let mut session_ptr: *mut OrtSession = std::ptr::null_mut();
    println!("Using Onnxruntime C API");
    let status = unsafe {
        g_ort.as_ref().unwrap().CreateSession.unwrap()(
            env_ptr,
            model_path.as_ptr(),
            session_options_ptr,
            &mut session_ptr,
        )
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(session_ptr, std::ptr::null_mut());

    //*************************************************************************
    // Gather model input layer info (node names, types, shape etc.)
    let mut allocator_ptr: *mut OrtAllocator = std::ptr::null_mut();
    let status = unsafe {
        g_ort
            .as_ref()
            .unwrap()
            .GetAllocatorWithDefaultOptions
            .unwrap()(&mut allocator_ptr)
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(allocator_ptr, std::ptr::null_mut());

    // Number of model input nodes
    let mut num_input_nodes: usize = 0;
    let status = unsafe {
        g_ort.as_ref().unwrap().SessionGetInputCount.unwrap()(session_ptr, &mut num_input_nodes)
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(num_input_nodes, 0);
    let mut input_node_names: Vec<&str> = Vec::new();
    let mut input_node_dims: Vec<i64> = Vec::new(); // FIXME simplify... this model has only 1 input node {1, 3, 224, 224}.
                                                    // Otherwise need vector<vector<>>

    // Iterate over all input nodes to gather names, types, shapes
    for i in 0..num_input_nodes {
        // Gather input node names
        let mut input_name: *mut i8 = std::ptr::null_mut();
        let status = unsafe {
            g_ort.as_ref().unwrap().SessionGetInputName.unwrap()(
                session_ptr,
                i,
                allocator_ptr,
                &mut input_name,
            )
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(input_name, std::ptr::null_mut());

        // WARNING: The C function SessionGetInputName allocates memory for the string.
        //          We cannot let Rust free that string, the C side must free the string.
        //          We thus convert the pointer to a string slice (&str).
        let input_name = char_p_to_str(input_name).unwrap();
        input_node_names.push(input_name);

        // Gather input node types
        let mut typeinfo_ptr: *mut OrtTypeInfo = std::ptr::null_mut();
        let status = unsafe {
            g_ort.as_ref().unwrap().SessionGetInputTypeInfo.unwrap()(
                session_ptr,
                i,
                &mut typeinfo_ptr,
            )
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(typeinfo_ptr, std::ptr::null_mut());

        let mut tensor_info_ptr: *const OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status = unsafe {
            g_ort.as_ref().unwrap().CastTypeInfoToTensorInfo.unwrap()(
                typeinfo_ptr,
                &mut tensor_info_ptr,
            )
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(tensor_info_ptr, std::ptr::null_mut());

        let mut type_: ONNXTensorElementDataType =
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        let status = unsafe {
            g_ort.as_ref().unwrap().GetTensorElementType.unwrap()(tensor_info_ptr, &mut type_)
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(
            type_,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
        );

        // Gather input shapes/dims
        let mut num_dims = 0;
        let status = unsafe {
            g_ort.as_ref().unwrap().GetDimensionsCount.unwrap()(tensor_info_ptr, &mut num_dims)
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(num_dims, 0);

        input_node_dims.resize_with(num_dims as usize, Default::default);
        let status = unsafe {
            g_ort.as_ref().unwrap().GetDimensions.unwrap()(
                tensor_info_ptr,
                input_node_dims.as_mut_ptr(),
                num_dims,
            )
        };
        CheckStatus(g_ort, status).unwrap();

        unsafe { g_ort.as_ref().unwrap().ReleaseTypeInfo.unwrap()(typeinfo_ptr) };
    }
    //*************************************************************************
    // Similar operations to get output node information.
    // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
    // OrtSessionGetOutputTypeInfo() as shown above.

    // FIXME : get output node names properly
    let output_node_names = &[config.io_shape.out_name];

    //*************************************************************************
    // Score the model using sample data, and inspect values
    let input_node_dims = config.io_shape.input_shape
        .iter()
        .map(|v| *v as i64)
        .collect::<Vec<_>>();

    // FIXME : use OrtGetTensorShapeElementCount() to get official size!
    let input_tensor_size: u64 = config.io_shape.input_shape
        .iter()
        .product();
    let input_tensor_size = input_tensor_size as usize;

    // initialize input data with values in [0.0, 1.0]
    let mut input_tensor_values: Vec<f32> = (0..input_tensor_size)
        .map(|i| (i as f32) / ((input_tensor_size + 1) as f32))
        .collect();

    // create input tensor object from data values
    let mut memory_info_ptr: *mut OrtMemoryInfo = std::ptr::null_mut();
    let status = unsafe {
        g_ort.as_ref().unwrap().CreateCpuMemoryInfo.unwrap()(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault,
            &mut memory_info_ptr,
        )
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(memory_info_ptr, std::ptr::null_mut());

    let mut input_tensor_ptr: *mut OrtValue = std::ptr::null_mut();
    let input_tensor_ptr_ptr: *mut *mut OrtValue = &mut input_tensor_ptr;
    let input_tensor_values_ptr: *mut std::ffi::c_void =
        input_tensor_values.as_mut_ptr() as *mut std::ffi::c_void;
    assert_ne!(input_tensor_values_ptr, std::ptr::null_mut());

    let shape: *const i64 = input_node_dims.as_ptr();
    assert_ne!(shape, std::ptr::null_mut());

    let status = unsafe {
        g_ort
            .as_ref()
            .unwrap()
            .CreateTensorWithDataAsOrtValue
            .unwrap()(
            memory_info_ptr,
            input_tensor_values_ptr,
            input_tensor_size * std::mem::size_of::<f32>(),
            shape,
            4,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            input_tensor_ptr_ptr,
        )
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(input_tensor_ptr, std::ptr::null_mut());

    let mut is_tensor = 0;
    let status =
        unsafe { g_ort.as_ref().unwrap().IsTensor.unwrap()(input_tensor_ptr, &mut is_tensor) };
    CheckStatus(g_ort, status).unwrap();
    assert_eq!(is_tensor, 1);

    let input_tensor_ptr2: *const OrtValue = input_tensor_ptr as *const OrtValue;
    let input_tensor_ptr3: *const *const OrtValue = &input_tensor_ptr2;

    unsafe { g_ort.as_ref().unwrap().ReleaseMemoryInfo.unwrap()(memory_info_ptr) };

    // Prepare input & output node names into pointers
    let input_node_names_cstring: Vec<std::ffi::CString> = input_node_names
        .into_iter()
        .map(|n| std::ffi::CString::new(n).unwrap())
        .collect();
    let input_node_names_ptr: Vec<*const i8> = input_node_names_cstring
        .into_iter()
        .map(|n| n.into_raw() as *const i8)
        .collect();
    let input_node_names_ptr_ptr: *const *const i8 = input_node_names_ptr.as_ptr();

    let output_node_names_cstring: Vec<std::ffi::CString> = output_node_names
        .into_iter()
        .map(|n| std::ffi::CString::new(n.clone()).unwrap())
        .collect();
    let output_node_names_ptr: Vec<*const i8> = output_node_names_cstring
        .iter()
        .map(|n| n.as_ptr() as *const i8)
        .collect();
    let output_node_names_ptr_ptr: *const *const i8 = output_node_names_ptr.as_ptr();

    // Run
    let run_options_ptr: *const OrtRunOptions = std::ptr::null();
    let mut output_tensor_ptr: *mut OrtValue = std::ptr::null_mut();
    let output_tensor_ptr_ptr: *mut *mut OrtValue = &mut output_tensor_ptr;

    let time = Instant::now();
    let status = unsafe {
        g_ort.as_ref().unwrap().Run.unwrap()(
            session_ptr,
            run_options_ptr,
            input_node_names_ptr_ptr,
            input_tensor_ptr3,
            1,
            output_node_names_ptr_ptr,
            1,
            output_tensor_ptr_ptr,
        )
    };
    println!("elapsed time {:?}", time.elapsed());
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(output_tensor_ptr, std::ptr::null_mut());

    // Get pointer to output tensor float values
    let mut is_tensor = 0;
    let status =
        unsafe { g_ort.as_ref().unwrap().IsTensor.unwrap()(output_tensor_ptr, &mut is_tensor) };
    CheckStatus(g_ort, status).unwrap();
    assert_eq!(is_tensor, 1);

    let mut floatarr: *mut f32 = std::ptr::null_mut();
    let floatarr_ptr: *mut *mut f32 = &mut floatarr;
    let floatarr_ptr_void: *mut *mut std::ffi::c_void = floatarr_ptr as *mut *mut std::ffi::c_void;
    let status = unsafe {
        g_ort.as_ref().unwrap().GetTensorMutableData.unwrap()(output_tensor_ptr, floatarr_ptr_void)
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(floatarr, std::ptr::null_mut());

    // Get output into vector
    let output_len: u64 = config.io_shape.output_shape
        .iter()
        .product();
    let output_len = output_len as usize;
    
    // let floatarr_vec: Vec<f32> = unsafe { Vec::from_raw_parts(floatarr, output_len, output_len) };
    // std::mem::forget(floatarr_vec);
    
    // copy bit-wise, allowing c api to free right amount of memory
    let mut output = Vec::with_capacity(output_len);
    unsafe { 
        copy(floatarr, output.as_mut_ptr(), output_len);
        output.set_len(output_len);
    }

    // Release the pointers
    unsafe { g_ort.as_ref().unwrap().ReleaseValue.unwrap()(output_tensor_ptr) };
    unsafe { g_ort.as_ref().unwrap().ReleaseValue.unwrap()(input_tensor_ptr) };
    unsafe { g_ort.as_ref().unwrap().ReleaseSession.unwrap()(session_ptr) };
    unsafe { g_ort.as_ref().unwrap().ReleaseSessionOptions.unwrap()(session_options_ptr) };
    unsafe { g_ort.as_ref().unwrap().ReleaseEnv.unwrap()(env_ptr) };

    println!("Done!");

    output
}

fn CheckStatus(g_ort: *const OrtApi, status: *const OrtStatus) -> Result<(), String> {
    if status != std::ptr::null() {
        let raw = unsafe { g_ort.as_ref().unwrap().GetErrorMessage.unwrap()(status) };
        Err(char_p_to_str(raw).unwrap().to_string())
    } else {
        Ok(())
    }
}

fn char_p_to_str<'a>(raw: *const i8) -> Result<&'a str, std::str::Utf8Error> {
    let c_str = unsafe { std::ffi::CStr::from_ptr(raw as *mut i8) };
    c_str.to_str()
}
