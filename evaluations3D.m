function evaluations3D(I_m, I_f, I_m_phi, I_f_phi_inv, phi, phi_inv, name_is,t_phi,t_phi_inv)
[nxm,nym,nzm] = size(I_m);
[nxf,nyf,nzf] = size(I_f);
phi_x = phi(:,:,:,1);
phi_y = phi(:,:,:,2);
phi_z = phi(:,:,:,3);

phi_x_inv = phi_inv(:,:,:,1);
phi_y_inv = phi_inv(:,:,:,2);
phi_z_inv = phi_inv(:,:,:,3);

mi_fwd = mutualInfo(I_m_phi, I_f);
I_m_on_I_f=imresize3(I_m, [nxf,nyf,nzf]);
mi_on_I_f = mutualInfo(I_f, I_m_on_I_f);
mi_fwd_r = (mi_fwd-mi_on_I_f)/mi_on_I_f;

mi_bwd = mutualInfo(I_m, I_f_phi_inv);
I_f_on_I_m=imresize3(I_f, [nxm,nym,nzm]);
mi_on_I_m = mutualInfo(I_f_on_I_m, I_m);
mi_bwd_r = (mi_bwd-mi_on_I_m)/mi_on_I_m;

thresh = graythresh(I_m_phi);
BW_I_m_phi = imbinarize(I_m_phi,thresh);

thresh = graythresh(I_f);
BW_I_f = imbinarize(I_f,thresh);

% j = jaccard(BW_I_m_phi,BW_I_f);
d = dice(BW_I_m_phi,BW_I_f);

ssd_original_on_I_f = sum((I_m_on_I_f - I_f).^2,"all");
r_fwd_on_I_f = sum((I_m_phi - I_f).^2,"all")/ssd_original_on_I_f;

ssd_original_on_I_m = sum((I_f_on_I_m - I_m).^2,"all");
r_bwd_on_I_m = sum((I_f_phi_inv - I_m).^2,"all")/ssd_original_on_I_m;


f_m=compute_JD3D(phi_x,phi_y, phi_z);
jdmax=max(f_m,[],"all");
jdmin=min(f_m,[],"all");


thresh = graythresh(I_f_phi_inv);
BW_I_f_psi = imbinarize(I_f_phi_inv,thresh);

thresh = graythresh(I_m);
BW_I_m = imbinarize(I_m,thresh);

% j2 = jaccard(BW_I_f_psi,BW_I_m);
d2 = dice(BW_I_f_psi,BW_I_m);


f_f=compute_JD3D(phi_x_inv, phi_y_inv, phi_z_inv);
jdmax2=max(f_f,[],"all");
jdmin2=min(f_f,[],"all");

filename = [name_is, 'evaluate_tabular','.txt'];
fileID = fopen(filename, 'w');
fprintf(fileID,['time_fwd ','Dice_fwd ', 'SSD_ratio_fwd ','MI_incr_%%_fwd ','JDmin_fwd ', 'JDmax_fwd ','time_inv ','Dice_inv ', 'SSD_ratio_inv ', 'MI_incr_%%_inv ','JDmin_inv ' ,'JDmax_inv ', ' \n']);
fprintf(fileID,[num2str(t_phi),'  ', num2str(d), '  ', num2str(r_fwd_on_I_f), '  ', num2str(mi_fwd_r*100),'   ', num2str(jdmin),'  ', num2str(jdmax), ' ',num2str(t_phi_inv), ' ',num2str(d2), '  ', num2str(r_bwd_on_I_m), '  ', num2str(mi_bwd_r*100),'  ', num2str(jdmin2),'  ', num2str(jdmax2), ' \n']);
fclose(fileID);
disp(['Evauation written to ' filename]);
end

