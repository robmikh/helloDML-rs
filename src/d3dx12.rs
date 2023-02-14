use windows::Win32::Graphics::{Direct3D12::{D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, D3D12_RESOURCE_DESC, D3D12_RESOURCE_FLAGS, D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12GraphicsCommandList, ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_SUBRESOURCE_DATA, D3D12_MEMCPY_DEST, D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, D3D12_TEXTURE_COPY_LOCATION_0, ID3D12Device, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_STATES, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES}, Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC}};

const SIZE_MAX: u64 = 0xffffffffffffffff;

#[repr(transparent)]
pub struct D3DX12HeapProperties(pub D3D12_HEAP_PROPERTIES);

impl D3DX12HeapProperties {
    pub fn new(ty: D3D12_HEAP_TYPE) -> Self {
        Self::new_with_masks(ty, 1, 1)
    }

    pub fn new_with_masks(ty: D3D12_HEAP_TYPE, creation_node_mask: u32, node_mask: u32) -> Self {
        Self (
            D3D12_HEAP_PROPERTIES {
                Type: ty,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: creation_node_mask,
                VisibleNodeMask: node_mask
            }
        )
    }
}

impl PartialEq for D3DX12HeapProperties {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

#[repr(transparent)]
pub struct D3DX12ResourceDesc(pub D3D12_RESOURCE_DESC);

impl D3DX12ResourceDesc {
    pub fn buffer(width: u64, flags: D3D12_RESOURCE_FLAGS) -> Self {
        Self::buffer_with_alignemnt(width, flags, 0)
    }
    pub fn buffer_with_alignemnt(width: u64, flags: D3D12_RESOURCE_FLAGS, alignment: u64) -> Self {
        Self (
            D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: alignment,
                Width: width,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            }
        )
    }
}

impl PartialEq for D3DX12ResourceDesc {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

#[repr(transparent)]
pub struct D3DX12TextureCopyLocation(pub D3D12_TEXTURE_COPY_LOCATION);

impl D3DX12TextureCopyLocation {
    pub fn new(resource: &ID3D12Resource, sub: u32) -> Self {
        Self(D3D12_TEXTURE_COPY_LOCATION { pResource: windows::core::ManuallyDrop::new(resource), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0{
            SubresourceIndex: sub
        } })
    }

    pub fn new_from_footprint(resource: &ID3D12Resource, footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT) -> Self {
        Self(D3D12_TEXTURE_COPY_LOCATION { pResource: windows::core::ManuallyDrop::new(resource), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0{
            PlacedFootprint: *footprint
        } })
    }
}

#[repr(transparent)]
pub struct D3DX12ResourceBarrier(pub D3D12_RESOURCE_BARRIER);

impl D3DX12ResourceBarrier {
    pub fn transition(resource: &ID3D12Resource, state_before: D3D12_RESOURCE_STATES, state_after: D3D12_RESOURCE_STATES) -> Self {
        Self(D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: windows::core::ManuallyDrop::new(resource),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: state_before,
                    StateAfter: state_after,
                }),
            },
        })
    }
}

pub fn update_subresources(
    command_list: &ID3D12GraphicsCommandList,
    destination_resource: &ID3D12Resource,
    intermediate: &ID3D12Resource,
    first_subresource: u32,
    num_subresources: u32,
    required_size: u64,
    layouts: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
    num_rows: &[u32],
    row_size_in_bytes: &[u64],
    src_data: &[D3D12_SUBRESOURCE_DATA],
) -> u64 {
    let temp: i64 = -1;

    // Minor validation
    let intermediate_desc = unsafe { intermediate.GetDesc() };
    let destination_desc = unsafe { destination_resource.GetDesc() };
    if intermediate_desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || 
        intermediate_desc.Width < required_size + layouts[0].Offset || 
        required_size > temp as u64 || 
        (destination_desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER && 
            (first_subresource != 0 || num_subresources != 1)) {
                return 0;
    }

    let mut data_ptr = std::ptr::null_mut();
    let result = unsafe { intermediate.Map(0, None, Some(&mut data_ptr)) };
    if result.is_err() {
        return 0;
    }

    for i in 0..num_subresources {
        if row_size_in_bytes[i as usize] > temp as u64 {
            return 0;
        }
        let dest_data = D3D12_MEMCPY_DEST {
            pData: unsafe { data_ptr.add(layouts[i as usize].Offset as usize) },
            RowPitch: layouts[i as usize].Footprint.RowPitch as usize,
            SlicePitch: layouts[i as usize].Footprint.RowPitch as usize * num_rows[i as usize] as usize,
        };
        memcpy_subresource(&dest_data, &src_data[i as usize], row_size_in_bytes[i as usize] as usize, num_rows[i as usize], layouts[i as usize].Footprint.Depth);
    }
    unsafe { intermediate.Unmap(0, None) }

    if destination_desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        unsafe { 
            command_list.CopyBufferRegion(destination_resource, 0, intermediate, layouts[0].Offset as u64, layouts[0].Footprint.Width as u64)
        }
    } else {
        for i in 0..num_subresources {
            let dest_location = D3DX12TextureCopyLocation::new(destination_resource, i + first_subresource);
            let src_location = D3DX12TextureCopyLocation::new_from_footprint(intermediate, &layouts[i as usize]);
            unsafe {
                command_list.CopyTextureRegion(&dest_location.0, 0, 0, 0, &src_location.0, None);
            }
        }
    } 

    required_size
}

pub fn update_subresource_heap(
    command_list: &ID3D12GraphicsCommandList,
    destination_resource: &ID3D12Resource,
    intermediate: &ID3D12Resource,
    intermediate_offset: u64,
    first_subresource: u32,
    num_subresources: u32,
    src_data: &[D3D12_SUBRESOURCE_DATA],
) -> u64 {
    let mut required_size = 0;
    let mem_to_alloc = (std::mem::size_of::<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>() + std::mem::size_of::<u32>() + std::mem::size_of::<u64>()) as u64 * num_subresources as u64;
    if mem_to_alloc > SIZE_MAX {
        return 0;
    }

    let mut mem = vec![0u8; mem_to_alloc as usize];
    let layouts = unsafe {
        let layout_ptr = mem.as_mut_ptr() as *mut D3D12_PLACED_SUBRESOURCE_FOOTPRINT;
        std::slice::from_raw_parts_mut(layout_ptr, num_subresources as usize)
    };
    let row_size_in_bytes = unsafe {
        std::slice::from_raw_parts_mut(layouts.as_mut_ptr().add(num_subresources as usize) as *mut u64, num_subresources as usize)
    };
    let num_rows = unsafe {
        std::slice::from_raw_parts_mut(row_size_in_bytes.as_mut_ptr().add(num_subresources as usize) as *mut u32, num_subresources as usize)
    };

    let desc = unsafe { destination_resource.GetDesc() };
    let d3d12_device: ID3D12Device = unsafe {
        let mut d3d12_device = None;
        destination_resource.GetDevice(&mut d3d12_device).unwrap();
        d3d12_device.unwrap()
    };
    unsafe {
        d3d12_device.GetCopyableFootprints(&desc, first_subresource, num_subresources, intermediate_offset, Some(layouts.as_mut_ptr()), Some(num_rows.as_mut_ptr()), Some(row_size_in_bytes.as_mut_ptr()), Some(&mut required_size as *mut _));
    }

    update_subresources(command_list, destination_resource, intermediate, first_subresource, num_subresources, required_size, layouts, num_rows, row_size_in_bytes, src_data)
}

pub fn memcpy_subresource(
    dest: &D3D12_MEMCPY_DEST,
    src: &D3D12_SUBRESOURCE_DATA,
    row_size_in_bytes: usize,
    num_rows: u32,
    num_slices: u32) {
        for z in 0..num_slices {
            let dest_ptr = unsafe { (dest.pData as *mut u8).add(dest.SlicePitch * z as usize) };
            let src_ptr = unsafe { (src.pData as *const u8).add(src.SlicePitch as usize * z as usize)};

            let dest_slice = unsafe { std::slice::from_raw_parts_mut(dest_ptr, row_size_in_bytes as usize) };
            let src_slice = unsafe { std::slice::from_raw_parts(src_ptr, row_size_in_bytes as usize) };
            for _ in 0..num_rows {
                dest_slice.copy_from_slice(src_slice);
            }
        }
    }