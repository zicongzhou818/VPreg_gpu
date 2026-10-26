clc
clear
close all

home_is = 'C:\Users\asonz\Documents\Old_Disk\ZicongZhouC\Study\Research\Codes\template_local_FCD_vpreg\VPreg_gpu';
ls_all = dir(fullfile(home_is, '**'));
ls_img_mov = ls_all(contains({ls_all.name},'img_mov.nii.gz'));
ls_img_mov_name = {ls_img_mov.name};
ls_img_fix = ls_all(contains({ls_all.name},'img_fix.nii.gz'));
ls_img_fix_name = {ls_img_fix.name};

aa=length(ls_img_mov_name);
bb=length(ls_img_fix_name);
for ii=1:aa
    for jj=1:bb
        % which_is=sprintf('%04d',(ii-1)*bb+jj);
        which_is = 'test';

        I_m_is = strcat(ls_img_mov(ii).folder,'\', ls_img_mov(ii).name);
        [~, name_mov, ~] = fileparts(I_m_is);
        [~, name_mov, ~] = fileparts(name_mov);
        I_m = single(niftiread(I_m_is));
        I_m_info = niftiinfo(I_m_is);
        I_m_info.Datatype = 'single';

        I_f_is = strcat(ls_img_fix(jj).folder, '\', ls_img_fix(jj).name);
        [~, name_fix, ~] = fileparts(I_f_is);
        [~, name_fix, ~] = fileparts(name_fix);
        I_f = single(niftiread(I_f_is));
        I_f_info = niftiinfo(I_f_is);
        I_f_info.Datatype = 'single';

        fprintf('\n');
        fprintf(['The ', which_is, '-th registration started.','\n']);

        name_is = strcat(ls_img_mov(ii).folder,'\',name_mov, '_VPreg_gpu_', which_is,'_');

        [phi, I_m_phi, phi_inv, I_f_phi_inv, ...
            toc_phi, toc_phi_inv] = VPreg_gpu(I_m, I_f);

        fprintf(['The ', which_is, '-th registration done.','\n']);

        niftiwrite(I_m_phi, [name_is,'Warped'], I_f_info, "Compressed", true);
        niftiwrite(phi, [name_is,'1Warp'], "Compressed", true);
        niftiwrite(I_f_phi_inv, [name_is, 'InverseWarped'], I_m_info, "Compressed", true);
        niftiwrite(phi_inv, [name_is, '1InverseWarp'], "Compressed", true);

        evaluations3D(I_m, I_f, I_m_phi, I_f_phi_inv, phi, phi_inv, name_is, toc_phi, toc_phi_inv);

    end
end

