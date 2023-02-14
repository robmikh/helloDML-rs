use windows::{Win32::Foundation::{HANDLE, CloseHandle}, core::Result};



#[repr(transparent)]
pub struct AutoCloseHandle(pub HANDLE);

impl AutoCloseHandle {
    pub fn close(&self) -> Result<()> {
        unsafe { CloseHandle(self.0).ok() }
    }
}

impl Drop for AutoCloseHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}