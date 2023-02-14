use windows::Win32::AI::MachineLearning::DirectML::{
    DML_TENSOR_DATA_TYPE, DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_DATA_TYPE_FLOAT32,
    DML_TENSOR_DATA_TYPE_FLOAT64, DML_TENSOR_DATA_TYPE_INT16, DML_TENSOR_DATA_TYPE_INT32,
    DML_TENSOR_DATA_TYPE_INT64, DML_TENSOR_DATA_TYPE_INT8, DML_TENSOR_DATA_TYPE_UINT16,
    DML_TENSOR_DATA_TYPE_UINT32, DML_TENSOR_DATA_TYPE_UINT64, DML_TENSOR_DATA_TYPE_UINT8,
};

pub fn dml_calc_buffer_tensor_size(
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
