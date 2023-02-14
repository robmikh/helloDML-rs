use windows::{
    core::Result,
    Win32::Foundation::{CloseHandle, HANDLE},
};

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
