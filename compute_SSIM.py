import numpy as np
from skimage.metrics import structural_similarity as ssim
def compute_mse(x,y):
    ret = np.average((x-y)**2)
    return ret
i=82
sza = [60.0,40.0,20.0,4.0]
vza = [60,30,15,0,-15,-30,-60]
s=3
v=3
for i in range(102):
    # load data
    image_data_dir = "/home/local/AD/ztushar1/Data/MultiView_Images_102_profiles"
    fname1 = image_data_dir+"/jet_100m_reflectance066_%05d_SZA_%02d_VZA_%02d.npy"%(i+1,sza[s],vza[v])
    fname2 = image_data_dir+"/jet_100m_reflectance066_%05d_SZA_%02d_VZA_%02d.npy"%(i+1,sza[s],vza[2])

    vza_0 = np.load(fname1)
    vza_15 = np.load(fname2)
    t=6
    temp=np.empty([t,t],dtype=float)
    for j in range (t):
        for k in range(t):
            temp[j,k]=ssim(vza_0[:144-t,0:144-t],vza_15[j:144-t+j,k:144-t+k])
            # ("index: ",j,k,ssim(vza_0[:144-t,0:144-t],vza_15[j:144-t+j,k:144-t+k]), "MSE: ",compute_mse(vza_0[:144-t,0:144-t],vza_15[j:144-t+j,k:144-t+k]))
    result = np.where(temp == np.amax(temp))
    print(result)



    
