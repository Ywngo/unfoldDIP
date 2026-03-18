# Existing code for worst-pixel probes

# Add the new observability probe 
def compute_observability_probes(X_obs, loss_hsi, loss_msi):
    # Compute total sensor loss
    total_loss = loss_hsi + loss_msi
    
    # Logic to calculate gradients at worst-pixel location in X_obs resolution
    sensor_grad = compute_gradient(total_loss, X_obs)  # Example of gradient computation
    worst_pixel_location = get_worst_pixel_location(total_loss)
    
    print(f"[Probe-Observability-SensorGrad] Gradient at worst pixel: {sensor_grad[worst_pixel_location]}")

# Keep existing worst-pixel probes
