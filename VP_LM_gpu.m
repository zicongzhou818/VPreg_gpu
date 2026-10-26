function [phi1_comp,phi2_comp,phi3_comp,phi_m1, phi_m2, phi_m3] = VP_LM_gpu(Nx, Ny, Nz, given_x, given_y, given_z, target_x, target_y, target_z)
%% Move data to GPU and initialization
% Convert inputs to gpuArray (if not already)
if ~isa(given_x, 'gpuArray'); given_x = gpuArray(given_x); end
if ~isa(given_y, 'gpuArray'); given_y = gpuArray(given_y); end
if ~isa(given_z, 'gpuArray'); given_z = gpuArray(given_z); end
if ~isa(target_x, 'gpuArray'); target_x = gpuArray(target_x); end
if ~isa(target_y, 'gpuArray'); target_y = gpuArray(target_y); end
if ~isa(target_z, 'gpuArray'); target_z = gpuArray(target_z); end

better = 1;
ei = 0;
ti = 0;
imax = 6.5e1;
ts = 1e-1;
ts_r = 5e-4;
ts_up = 1.1;
ts_dn = 0.9;

r = 1;
r_tol = 1e-9;

%% preset new deformation based on control functions that are corresponsive to the identity map
[X, Y, Z] = ndgrid(1:Nx, 1:Ny, 1:Nz);
% move grids to GPU
X = gpuArray(X);
Y = gpuArray(Y);
Z = gpuArray(Z);

phi_m1 = X;
phi_m2 = Y;
phi_m3 = Z;
% preallocate next-step arrays on GPU
phi_m1n = phi_m1;
phi_m2n = phi_m2;
phi_m3n = phi_m3;

%% reading given mesh (already gpuArray)
phi_o1 = given_x;
phi_o2 = given_y;
phi_o3 = given_z;

%% initial composed phi is the given mesh
phi1_comp = phi_o1;
phi2_comp = phi_o2;
phi3_comp = phi_o3;

%% compute ssd (on GPU)
ssd_initial = sum((phi1_comp - target_x).^2 + (phi2_comp - target_y).^2 + (phi3_comp - target_z).^2, "all");
ssd_old = ssd_initial;

%% main loop (all on GPU)
while ti < imax && ts > ts_r && r > r_tol
    ti = ti + 1;
    if better
        ei = ei + 1;
        % Simple gradient (as in original)
        Gx = (phi1_comp - target_x);
        Gy = (phi2_comp - target_y);
        Gz = (phi3_comp - target_z);
    end

    % Update interior voxels (indices remain valid on GPU arrays)
    phi_m1n(2:Nx-1, 2:Ny-1, 2:Nz-1) = phi_m1(2:Nx-1, 2:Ny-1, 2:Nz-1) - ts * Gx(2:Nx-1, 2:Ny-1, 2:Nz-1);
    phi_m2n(2:Nx-1, 2:Ny-1, 2:Nz-1) = phi_m2(2:Nx-1, 2:Ny-1, 2:Nz-1) - ts * Gy(2:Nx-1, 2:Ny-1, 2:Nz-1);
    phi_m3n(2:Nx-1, 2:Ny-1, 2:Nz-1) = phi_m3(2:Nx-1, 2:Ny-1, 2:Nz-1) - ts * Gz(2:Nx-1, 2:Ny-1, 2:Nz-1);

    % Interpolate given mesh at new phi_mn positions.
    % Assumes gpu_speed_up_interp accepts gpuArray inputs and returns gpuArray.
    phi1_compn = interpn(X, Y, Z, phi_o1, phi_m1n, phi_m2n, phi_m3n,'LINEAR');
    phi2_compn = interpn(X, Y, Z, phi_o2, phi_m1n, phi_m2n, phi_m3n,'LINEAR');
    phi3_compn = interpn(X, Y, Z, phi_o3, phi_m1n, phi_m2n, phi_m3n,'LINEAR');

    % Compute new SSD on GPU
    ssd_new = sum((phi1_compn - target_x).^2 + (phi2_compn - target_y).^2 + (phi3_compn - target_z).^2, "all");
    r = ssd_new / ssd_initial;

    if ssd_new < ssd_old
        ts = ts * ts_up;
        % accept step (assign GPU arrays)
        phi_m1 = phi_m1n;
        phi_m2 = phi_m2n;
        phi_m3 = phi_m3n;
        phi1_comp = phi1_compn;
        phi2_comp = phi2_compn;
        phi3_comp = phi3_compn;
        ssd_old = ssd_new;
        better = 1;
    else
        % reject step, decrease step-size; phi_m* remain previous
        better = 0;
        ts = ts * ts_dn;
    end
    % disp(['ts:', num2str(ts), ' r:', num2str(r), ' ei/ti:', num2str(ei), '/', num2str(ti)]);

end

% Display convergence info (gather scalars to host for printing)
try
    ts_disp = gather(ts);
    r_disp = gather(ssd_new / ssd_initial);
catch
    ts_disp = ts;
    r_disp = ssd_new / ssd_initial;
end
disp(['inverse found ts:', num2str(ts_disp), ' r:', num2str(r_disp), ' ei/ti:', num2str(ei), '/', num2str(ti)]);

% Gather outputs to CPU arrays before returning (optional)
phi1_comp = gather(phi1_comp);
phi2_comp = gather(phi2_comp);
phi3_comp = gather(phi3_comp);
phi_m1 = gather(phi_m1);
phi_m2 = gather(phi_m2);
phi_m3 = gather(phi_m3);

end
