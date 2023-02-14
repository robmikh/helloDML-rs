mod d3dx12;
mod handle;

use d3dx12::{D3DX12HeapProperties, D3DX12ResourceDesc};
use handle::AutoCloseHandle;
use windows::{
    core::{Interface, Result},
    Win32::{
        Graphics::{
            Direct3D::D3D_FEATURE_LEVEL_11_0,
            Direct3D12::{
                D3D12CreateDevice, D3D12GetDebugInterface, ID3D12CommandAllocator,
                ID3D12CommandQueue, ID3D12Debug, ID3D12DescriptorHeap, ID3D12Device, ID3D12Fence,
                ID3D12GraphicsCommandList, ID3D12Resource, D3D12_COMMAND_LIST_TYPE_DIRECT,
                D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
                D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_FENCE_FLAG_NONE,
                D3D12_HEAP_FLAG_NONE, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_READBACK,
                D3D12_HEAP_TYPE_UPLOAD, D3D12_RANGE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_SUBRESOURCE_DATA,
            },
            Dxgi::{
                CreateDXGIFactory1, IDXGIFactory4, DXGI_ERROR_SDK_COMPONENT_MISSING,
                DXGI_ERROR_UNSUPPORTED,
            },
        },
        System::{
            Threading::{CreateEventW, WaitForSingleObjectEx},
            WindowsProgramming::INFINITE,
        },
        AI::MachineLearning::DirectML::{
            DMLCreateDevice, IDMLBindingTable, IDMLCommandRecorder, IDMLCompiledOperator,
            IDMLDevice, IDMLOperator, IDMLOperatorInitializer, DML_BINDING_DESC,
            DML_BINDING_TABLE_DESC, DML_BINDING_TYPE_BUFFER, DML_BUFFER_BINDING,
            DML_BUFFER_TENSOR_DESC, DML_CREATE_DEVICE_FLAG_DEBUG, DML_CREATE_DEVICE_FLAG_NONE,
            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC, DML_EXECUTION_FLAG_NONE, DML_OPERATOR_DESC,
            DML_OPERATOR_ELEMENT_WISE_IDENTITY, DML_TENSOR_DATA_TYPE, DML_TENSOR_DATA_TYPE_FLOAT16,
            DML_TENSOR_DATA_TYPE_FLOAT32, DML_TENSOR_DATA_TYPE_FLOAT64, DML_TENSOR_DATA_TYPE_INT16,
            DML_TENSOR_DATA_TYPE_INT32, DML_TENSOR_DATA_TYPE_INT64, DML_TENSOR_DATA_TYPE_INT8,
            DML_TENSOR_DATA_TYPE_UINT16, DML_TENSOR_DATA_TYPE_UINT32, DML_TENSOR_DATA_TYPE_UINT64,
            DML_TENSOR_DATA_TYPE_UINT8, DML_TENSOR_DESC, DML_TENSOR_FLAG_NONE,
            DML_TENSOR_TYPE_BUFFER,
        },
    },
};

use crate::d3dx12::{update_subresource_heap, D3DX12ResourceBarrier};

const TENSOR_SIZES: [u32; 4] = [1, 2, 3, 4];
const TENSOR_ELEMENT_COUNT: u32 =
    TENSOR_SIZES[0] * TENSOR_SIZES[1] * TENSOR_SIZES[2] * TENSOR_SIZES[3];

fn main() -> Result<()> {
    // Setup Direct3D12
    let (d3d12_device, command_queue, command_allocator, command_list) = init_d3d12()?;

    // Create the DirectML device

    let dml_create_device_flags = if cfg!(feature = "dxdebug") {
        DML_CREATE_DEVICE_FLAG_DEBUG
    } else {
        DML_CREATE_DEVICE_FLAG_NONE
    };

    let dml_device: IDMLDevice = unsafe {
        let mut dml_device = None;
        DMLCreateDevice(&d3d12_device, dml_create_device_flags, &mut dml_device)?;
        dml_device.unwrap()
    };

    let dml_buffer_tensor_desc = {
        let data_type = DML_TENSOR_DATA_TYPE_FLOAT32;
        let dimension_count = TENSOR_SIZES.len() as u32;

        DML_BUFFER_TENSOR_DESC {
            DataType: data_type,
            Flags: DML_TENSOR_FLAG_NONE,
            DimensionCount: dimension_count,
            Sizes: TENSOR_SIZES.as_ptr(),
            Strides: std::ptr::null(),
            TotalTensorSizeInBytes: dml_calc_buffer_tensor_size(
                data_type,
                dimension_count,
                &TENSOR_SIZES,
                None,
            ),
            ..Default::default()
        }
    };

    let dml_operator: IDMLOperator = unsafe {
        // Create DirectML operator(s). Operators represent abstract functions such as "multiply", "reduce", "convolution", or even
        // compound operations such as recurrent neural nets. This example creates an instance of the Identity operator,
        // which applies the function f(x) = x for all elements in a tensor.

        let dml_tensor_desc = DML_TENSOR_DESC {
            Type: DML_TENSOR_TYPE_BUFFER,
            Desc: &dml_buffer_tensor_desc as *const _ as *const _,
            ..Default::default()
        };

        let dml_identity_operator_desc = DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC {
            InputTensor: &dml_tensor_desc,
            OutputTensor: &dml_tensor_desc, // Input and output tensors have smae size/type
            ..Default::default()
        };

        // Like Direct3D 12, these DESC structs don't need to be long-lived. This means, for example, that it's safe to place
        // the DML_OPERATOR_DESC (and all the subobjects it points to) on the stack, since they're no longer needed after
        // CreateOperator returns.
        let dml_operator_desc = DML_OPERATOR_DESC {
            Type: DML_OPERATOR_ELEMENT_WISE_IDENTITY,
            Desc: &dml_identity_operator_desc as *const _ as *const _,
            ..Default::default()
        };

        let mut dml_operator = None;
        dml_device.CreateOperator(&dml_operator_desc, &mut dml_operator)?;
        dml_operator.unwrap()
    };

    // Compile the operator into an object that can be dispatched to the GPU. In this step, DirectML performs operator
    // fusion and just-in-time (JIT) compilation of shader bytecode, then compiles it into a Direct3D 12 pipeline state object (PSO).
    // The resulting compiled operator is a baked, optimized form of an operator suitable for execution on the GPU.

    let dml_compiled_operator: IDMLCompiledOperator = unsafe {
        let mut dml_compiled_operator = None;
        dml_device.CompileOperator(
            &dml_operator,
            DML_EXECUTION_FLAG_NONE,
            &mut dml_compiled_operator,
        )?;
        dml_compiled_operator.unwrap()
    };

    // 24 elements * 4 == 96 bytes.
    let tensor_buffer_size = dml_buffer_tensor_desc.TotalTensorSizeInBytes;

    let dml_operator_initializer: IDMLOperatorInitializer =
        unsafe { dml_device.CreateOperatorInitializer(Some(&[dml_compiled_operator.clone()]))? };

    // Query the operator for the required size (in descriptors) of its binding table.
    // You need to initialize an operator exactly once before it can be executed, and
    // the two stages require different numbers of descriptors for binding. For simplicity,
    // we create a single descriptor heap that's large enough to satisfy them both.
    let initialize_binding_properties = unsafe { dml_operator_initializer.GetBindingProperties() };
    let execute_binding_properties = unsafe { dml_compiled_operator.GetBindingProperties() };
    let descriptor_count = initialize_binding_properties
        .RequiredDescriptorCount
        .max(execute_binding_properties.RequiredDescriptorCount);

    // Create descriptor heaps
    let descriptor_heap: ID3D12DescriptorHeap = unsafe {
        let descriptor_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: descriptor_count,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            ..Default::default()
        };
        d3d12_device.CreateDescriptorHeap(&descriptor_heap_desc)?
    };

    // Set the descriptor heap(s).
    unsafe {
        command_list.SetDescriptorHeaps(&[descriptor_heap.clone()]);
    }

    // Create a binding table over the descriptor heap we just created.
    let dispatchable = dml_operator_initializer.cast()?;
    let mut dml_binding_table_desc = DML_BINDING_TABLE_DESC {
        Dispatchable: windows::core::ManuallyDrop::new(&dispatchable),
        CPUDescriptorHandle: unsafe { descriptor_heap.GetCPUDescriptorHandleForHeapStart() },
        GPUDescriptorHandle: unsafe { descriptor_heap.GetGPUDescriptorHandleForHeapStart() },
        SizeInDescriptors: descriptor_count,
        ..Default::default()
    };
    let dml_binding_table: IDMLBindingTable =
        unsafe { dml_device.CreateBindingTable(Some(&dml_binding_table_desc))? };

    // Create the temporary and persistent resources that are necessary for executing an operator.

    // The temporary resource is scratch memory (used internally by DirectML), whose contents you don't need to define.
    // The persistent resource is long-lived, and you need to initialize it using the IDMLOperatorInitializer.

    let temporary_resource_size = initialize_binding_properties
        .TemporaryResourceSize
        .max(execute_binding_properties.TemporaryResourceSize);
    let persistent_resource_size = execute_binding_properties.PersistentResourceSize;

    // Bind and initialize the operator on the GPU.

    let temporary_buffer: Option<ID3D12Resource> = unsafe {
        if temporary_resource_size != 0 {
            let heap_properties = D3DX12HeapProperties::new(D3D12_HEAP_TYPE_DEFAULT);
            let buffer_desc = D3DX12ResourceDesc::buffer(
                temporary_resource_size,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            );

            let mut temporary_buffer = None;
            d3d12_device.CreateCommittedResource(
                &heap_properties.0,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc.0,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut temporary_buffer,
            )?;
            let temporary_buffer = temporary_buffer.unwrap();

            if initialize_binding_properties.TemporaryResourceSize != 0 {
                let buffer_binding = DML_BUFFER_BINDING {
                    Buffer: windows::core::ManuallyDrop::new(&temporary_buffer),
                    Offset: 0,
                    SizeInBytes: temporary_resource_size,
                };
                let binding_desc = DML_BINDING_DESC {
                    Type: DML_BINDING_TYPE_BUFFER,
                    Desc: &buffer_binding as *const _ as *const _,
                };
                dml_binding_table.BindTemporaryResource(Some(&binding_desc))
            }

            Some(temporary_buffer)
        } else {
            None
        }
    };

    let persistent_buffer: Option<ID3D12Resource> = unsafe {
        if persistent_resource_size != 0 {
            let heap_properties = D3DX12HeapProperties::new(D3D12_HEAP_TYPE_DEFAULT);
            let buffer_desc =
                D3DX12ResourceDesc::buffer(temporary_resource_size, D3D12_RESOURCE_FLAG_NONE);

            let mut persistent_buffer = None;
            d3d12_device.CreateCommittedResource(
                &heap_properties.0,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc.0,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut persistent_buffer,
            )?;
            let persistent_buffer = persistent_buffer.unwrap();

            // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
            let buffer_binding = DML_BUFFER_BINDING {
                Buffer: windows::core::ManuallyDrop::new(&persistent_buffer),
                Offset: 0,
                SizeInBytes: persistent_resource_size,
            };
            let binding_desc = DML_BINDING_DESC {
                Type: DML_BINDING_TYPE_BUFFER,
                Desc: &buffer_binding as *const _ as *const _,
            };
            dml_binding_table.BindOutputs(Some(&[binding_desc]));

            Some(persistent_buffer)
        } else {
            None
        }
    };

    // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
    let dml_command_recorder: IDMLCommandRecorder = unsafe { dml_device.CreateCommandRecorder()? };

    // Record execution of the operator initializer.
    unsafe {
        dml_command_recorder.RecordDispatch(
            &command_list,
            &dml_operator_initializer,
            &dml_binding_table,
        )
    }

    // Close the Direct3D 12 command list, and submit it for execution as you would any other command list. You could
    // in principle record the execution into the same command list as the initialization, but you need only to Initialize
    // once, and typically you want to Execute an operator more frequently than that.
    close_execute_reset_wait(
        &d3d12_device,
        &command_queue,
        &command_allocator,
        &command_list,
    )?;

    //
    // Bind and execute the operator on the GPU.
    //

    unsafe {
        command_list.SetDescriptorHeaps(&[descriptor_heap.clone()]);
    }

    // Reset the binding table to bind for the operator we want to execute (it was previously used to bind for the
    // initializer).

    let dispatchable = dml_compiled_operator.cast()?;
    dml_binding_table_desc.Dispatchable = windows::core::ManuallyDrop::new(&dispatchable);
    unsafe { dml_binding_table.Reset(Some(&dml_binding_table_desc))? }

    if let Some(temporary_buffer) = &temporary_buffer {
        let buffer_binding = DML_BUFFER_BINDING {
            Buffer: windows::core::ManuallyDrop::new(&temporary_buffer),
            Offset: 0,
            SizeInBytes: temporary_resource_size,
        };
        let binding_desc = DML_BINDING_DESC {
            Type: DML_BINDING_TYPE_BUFFER,
            Desc: &buffer_binding as *const _ as *const _,
        };
        unsafe { dml_binding_table.BindTemporaryResource(Some(&binding_desc)) }
    }

    if let Some(persistent_buffer) = &persistent_buffer {
        let buffer_binding = DML_BUFFER_BINDING {
            Buffer: windows::core::ManuallyDrop::new(&persistent_buffer),
            Offset: 0,
            SizeInBytes: persistent_resource_size,
        };
        let binding_desc = DML_BINDING_DESC {
            Type: DML_BINDING_TYPE_BUFFER,
            Desc: &buffer_binding as *const _ as *const _,
        };
        unsafe { dml_binding_table.BindOutputs(Some(&[binding_desc])) }
    }

    // Create tensor buffers for upload/input/output/readback of the tensor elements.

    let upload_buffer: ID3D12Resource = unsafe {
        let heap_properties = D3DX12HeapProperties::new(D3D12_HEAP_TYPE_UPLOAD);
        let buffer_desc = D3DX12ResourceDesc::buffer(tensor_buffer_size, D3D12_RESOURCE_FLAG_NONE);

        let mut upload_buffer = None;
        d3d12_device.CreateCommittedResource(
            &heap_properties.0,
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc.0,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut upload_buffer,
        )?;
        upload_buffer.unwrap()
    };

    let input_buffer: ID3D12Resource = unsafe {
        let heap_properties = D3DX12HeapProperties::new(D3D12_HEAP_TYPE_DEFAULT);
        let buffer_desc = D3DX12ResourceDesc::buffer(
            tensor_buffer_size,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        );

        let mut input_buffer = None;
        d3d12_device.CreateCommittedResource(
            &heap_properties.0,
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc.0,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut input_buffer,
        )?;
        input_buffer.unwrap()
    };

    let input_tensor_element_array = vec![1.618f32; TENSOR_ELEMENT_COUNT as usize];
    print!("input tensor: ");
    for element in &input_tensor_element_array {
        print!("{:.2} ", element);
    }
    println!("");

    let tensor_subresource_data = D3D12_SUBRESOURCE_DATA {
        pData: input_tensor_element_array.as_ptr() as *const _,
        RowPitch: tensor_buffer_size as isize,
        SlicePitch: tensor_buffer_size as isize,
        ..Default::default()
    };

    // Upload the input tensor to the GPU.
    update_subresource_heap(
        &command_list,
        &input_buffer,
        &upload_buffer,
        0,
        0,
        1,
        &[tensor_subresource_data],
    );

    unsafe {
        let transition_barrier = D3DX12ResourceBarrier::transition(
            &input_buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        command_list.ResourceBarrier(&[transition_barrier.0])
    }

    let input_buffer_binding = DML_BUFFER_BINDING {
        Buffer: windows::core::ManuallyDrop::new(&input_buffer),
        Offset: 0,
        SizeInBytes: tensor_buffer_size,
    };
    let input_binding_desc = DML_BINDING_DESC {
        Type: DML_BINDING_TYPE_BUFFER,
        Desc: &input_buffer_binding as *const _ as *const _,
    };
    unsafe { dml_binding_table.BindInputs(Some(&[input_binding_desc])) }

    let output_buffer: ID3D12Resource = unsafe {
        let heap_properties = D3DX12HeapProperties::new(D3D12_HEAP_TYPE_DEFAULT);
        let buffer_desc = D3DX12ResourceDesc::buffer(
            tensor_buffer_size,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        );

        let mut output_buffer = None;
        d3d12_device.CreateCommittedResource(
            &heap_properties.0,
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc.0,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
            &mut output_buffer,
        )?;
        output_buffer.unwrap()
    };

    let output_buffer_binding = DML_BUFFER_BINDING {
        Buffer: windows::core::ManuallyDrop::new(&output_buffer),
        Offset: 0,
        SizeInBytes: tensor_buffer_size,
    };
    let output_binding_desc = DML_BINDING_DESC {
        Type: DML_BINDING_TYPE_BUFFER,
        Desc: &output_buffer_binding as *const _ as *const _,
    };
    unsafe { dml_binding_table.BindOutputs(Some(&[output_binding_desc])) }

    // Record execution of the compiled operator
    unsafe {
        dml_command_recorder.RecordDispatch(
            &command_list,
            &dml_compiled_operator,
            &dml_binding_table,
        )
    }

    close_execute_reset_wait(
        &d3d12_device,
        &command_queue,
        &command_allocator,
        &command_list,
    )?;

    // The output buffer now contains the result of the identity operator,
    // so read it back if you want the CPU to access it.

    let readback_buffer: ID3D12Resource = unsafe {
        let heap_properties = D3DX12HeapProperties::new(D3D12_HEAP_TYPE_READBACK);
        let buffer_desc = D3DX12ResourceDesc::buffer(tensor_buffer_size, D3D12_RESOURCE_FLAG_NONE);

        let mut readback_buffer = None;
        d3d12_device.CreateCommittedResource(
            &heap_properties.0,
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc.0,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut readback_buffer,
        )?;
        readback_buffer.unwrap()
    };

    unsafe {
        let transition_barrier = D3DX12ResourceBarrier::transition(
            &output_buffer,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        command_list.ResourceBarrier(&[transition_barrier.0]);

        command_list.CopyResource(&readback_buffer, &output_buffer);
    }

    close_execute_reset_wait(
        &d3d12_device,
        &command_queue,
        &command_allocator,
        &command_list,
    )?;

    let tensor_buffer_range = D3D12_RANGE {
        Begin: 0,
        End: tensor_buffer_size as usize,
    };
    let mut output_buffer_data: *mut f32 = unsafe {
        let mut output_buffer_data = std::ptr::null_mut();
        readback_buffer.Map(
            0,
            Some(&tensor_buffer_range),
            Some(&mut output_buffer_data as *mut _ as *mut _),
        )?;
        output_buffer_data
    };

    print!("output tensor: ");
    for _ in 0..TENSOR_ELEMENT_COUNT {
        unsafe {
            print!("{:.2} ", *output_buffer_data);
            output_buffer_data = output_buffer_data.add(1);
        }
    }
    println!("");

    let empty_range = D3D12_RANGE::default();
    unsafe { readback_buffer.Unmap(0, Some(&empty_range)) }

    Ok(())
}

fn init_d3d12() -> Result<(
    ID3D12Device,
    ID3D12CommandQueue,
    ID3D12CommandAllocator,
    ID3D12GraphicsCommandList,
)> {
    if cfg!(feature = "dxdebug") {
        let d3d12_debug: ID3D12Debug = unsafe {
            let mut d3d12_debug = None;
            let debug_result = D3D12GetDebugInterface(&mut d3d12_debug);
            if debug_result.is_err() {
                return Err(DXGI_ERROR_SDK_COMPONENT_MISSING.into());
            }
            d3d12_debug.unwrap()
        };
        unsafe { d3d12_debug.EnableDebugLayer() };
    }

    let dxgi_factory: IDXGIFactory4 = unsafe { CreateDXGIFactory1()? };

    let d3d12_device: ID3D12Device = {
        let mut d3d12_device = None;

        let mut adapter_index = 0;
        while let Ok(adapter) = unsafe { dxgi_factory.EnumAdapters(adapter_index) } {
            let result =
                unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut d3d12_device) };
            if let Err(error) = result {
                if error.code() != DXGI_ERROR_UNSUPPORTED {
                    return Err(error);
                }
            } else {
                break;
            }
            adapter_index += 1;
        }

        if d3d12_device.is_some() {
            d3d12_device.unwrap()
        } else {
            return Err(DXGI_ERROR_UNSUPPORTED.into());
        }
    };

    let command_queue: ID3D12CommandQueue = unsafe {
        let desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            ..Default::default()
        };
        d3d12_device.CreateCommandQueue(&desc)?
    };

    let command_allocator: ID3D12CommandAllocator =
        unsafe { d3d12_device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)? };

    let command_list: ID3D12GraphicsCommandList = unsafe {
        d3d12_device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            &command_allocator,
            None,
        )?
    };

    Ok((d3d12_device, command_queue, command_allocator, command_list))
}

fn close_execute_reset_wait(
    d3d12_device: &ID3D12Device,
    command_queue: &ID3D12CommandQueue,
    command_allocator: &ID3D12CommandAllocator,
    command_list: &ID3D12GraphicsCommandList,
) -> Result<()> {
    unsafe {
        command_list.Close()?;

        command_queue.ExecuteCommandLists(&[command_list.cast()?]);

        let d3d12_fence: ID3D12Fence = d3d12_device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;

        let handle = AutoCloseHandle(CreateEventW(None, true, false, None)?);
        d3d12_fence.SetEventOnCompletion(1, handle.0)?;
        command_queue.Signal(&d3d12_fence, 1)?;
        WaitForSingleObjectEx(handle.0, INFINITE, false).ok()?;

        command_allocator.Reset()?;
        command_list.Reset(command_allocator, None)?;
    }

    Ok(())
}

fn dml_calc_buffer_tensor_size(
    data_type: DML_TENSOR_DATA_TYPE,
    dimension_count: u32,
    sizes: &[u32],
    strides: Option<&[u32]>,
) -> u64 {
    assert!(sizes.len() as u32 == dimension_count);
    if let Some(strides) = strides.as_ref() {
        assert!(strides.len() as u32 == dimension_count);
    };

    let element_size_in_bytes = match data_type {
        DML_TENSOR_DATA_TYPE_FLOAT32 | DML_TENSOR_DATA_TYPE_UINT32 | DML_TENSOR_DATA_TYPE_INT32 => {
            4
        }
        DML_TENSOR_DATA_TYPE_FLOAT16 | DML_TENSOR_DATA_TYPE_UINT16 | DML_TENSOR_DATA_TYPE_INT16 => {
            2
        }
        DML_TENSOR_DATA_TYPE_UINT8 | DML_TENSOR_DATA_TYPE_INT8 => 1,
        DML_TENSOR_DATA_TYPE_FLOAT64 | DML_TENSOR_DATA_TYPE_UINT64 | DML_TENSOR_DATA_TYPE_INT64 => {
            8
        }
        _ => 0, // Invalid data type
    };

    let mut minimum_implied_size_in_bytes;
    if let Some(strides) = strides {
        let mut index_of_last_element = 0;
        for i in 0..dimension_count {
            index_of_last_element += (sizes[i as usize] - 1) * strides[i as usize];
        }
        minimum_implied_size_in_bytes = (index_of_last_element as u64 + 1) * element_size_in_bytes;
    } else {
        minimum_implied_size_in_bytes = sizes[0] as u64;
        for i in 1..dimension_count {
            minimum_implied_size_in_bytes *= sizes[i as usize] as u64;
        }
        minimum_implied_size_in_bytes *= element_size_in_bytes;
    }

    // Round up to the nearest 4 bytes
    minimum_implied_size_in_bytes = (minimum_implied_size_in_bytes + 3) & !3;

    minimum_implied_size_in_bytes
}
