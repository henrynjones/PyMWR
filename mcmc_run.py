import mwr
import mrcfile
import torch

sigma_noise = 0.05
plotFlag    = 1 # % set to 1 to observe processing in real time

#% Launch processing:
Vin = torch.load("./emb_sample.pt")
wedge = torch.load("./emb_mask.pt")
Vin = Vin[Vin.shape[0] //2 - 50 : Vin.shape[0] //2 + 50,
            Vin.shape[1] //2 - 50: Vin.shape[1]//2 + 50,
            Vin.shape[2] //2 - 50: Vin.shape[2] //2 + 50]

wedge = wedge[wedge.shape[0] //2 - 50 : wedge.shape[0] //2 + 50,
            wedge.shape[1] //2 - 50: wedge.shape[1]//2 + 50,
            wedge.shape[2] //2 - 50: wedge.shape[2] //2 + 50]
Vout = mwr.mwr(Vin, sigma_noise, wedge, plotFlag, T=10)

torch.save(Vout, "./mwr.pt")
