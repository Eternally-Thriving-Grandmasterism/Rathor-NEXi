#![cfg_attr(
  all(not(debug_assertions), target_os = "windows"),
  windows_subsystem = "windows"
)]

use tauri_plugin_updater::UpdaterExt;

fn main() {
  tauri::Builder::default()
    .plugin(tauri_plugin_updater::Builder::new().build())
    .setup(|app| {
      let handle = app.handle().clone();
      tauri::async_runtime::spawn(async move {
        if let Some(update) = handle.updater().check().await.unwrap_or(None) {
          if update.is_update_available() {
            // Optional: Show dialog or auto-download/install
            let _ = update.download_and_install(|_chunk, _total| {}, || {}).await;
          }
        }
      });
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("Mercy thunder error while running Rathor Lattice");
}
