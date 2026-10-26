function [JD_phi]=compute_JD3D(phi_1, phi_2, phi_3)
[Phi1y,Phi1x,Phi1z]=gradient(phi_1,1);
[Phi2y,Phi2x,Phi2z]=gradient(phi_2,1);
[Phi3y,Phi3x,Phi3z]=gradient(phi_3,1);
JD_phi=Phi1x.*(Phi2y.*Phi3z-Phi2z.*Phi3y)...
    -Phi2x.*(Phi1y .* Phi3z - Phi1z .* Phi3y)...
    +Phi3x.*(Phi1y.*Phi2z-Phi1z.*Phi2y);
end