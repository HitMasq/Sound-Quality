import numpy as np
import pandas as pd

freq_low = np.array(
    [0, 20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000])
freq_high = np.array(
    [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500])
freq_c = np.array(
    [10, 50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500, 10500, 13500])
z_bark = [0]        # Bark serial number
for f in freq_c[1:]:
    z = 13 * np.arctan(0.00076*f) + 3.5 * np.arctan(f/7500)**2
    z_bark.append(int(round(z))+1)

critical_band = np.array([freq_c, freq_low, freq_high, freq_high-freq_low]).T

columns = ['freq_center', 'freq_low', 'freq_high', 'bandwidth']
df = pd.DataFrame(critical_band, index=z_bark, columns=columns)
df.to_excel("save/critical_band.xlsx")
