# Calculate diffraction patterns of arbitrary apertures

import numpy as np
from numpy import fft
from numpy import pi,exp
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from mpl_toolkits.mplot3d import Axes3D 


## function definitions
def ellipse(x,y,a=1,b=2):
    return ((x/a)**2 + (y/b)**2) < 1
    # set a = b for circle

def slit(x,y,a=1,b=2):
    return (np.abs(2*x/a) < 1) * (np.abs(2*y/b) < 1)


## initial data
# all lengths measured in wavelength units

E_0 = 1        # let's use unit amplitude for the wave
k = 2*pi       # because lambda = 1    
d = 5e5        # distance from input plane to output plane

x = np.linspace(-3,3,2000)
y = np.linspace(-3,3,2000)
[Xin,Yin] = np.meshgrid(x,y)
[Xout,Yout] = np.meshgrid(x,y)

k_x = fft.fftshift(fft.fftfreq(len(x)))
k_y = fft.fftshift(fft.fftfreq(len(y)))
[k_X, k_Y] = np.meshgrid(k_x,k_y)

## propagation through distance d

# ---------- Step 0: define input field ---------------
# (uncomment one of the following)
#

# single slit
E_in = E_0 * slit(Xin,Yin,0.1,0.3) 

# double slit
#E_in = E_0 * ( slit(Xin,Yin,0.05,0.3)  + slit(Xin-0.15,Yin,0.05,0.3) )

# grating
#E_in = 0
#for i in range(-50, 50):
    #E_in += slit(Xin-i*0.15,Yin,0.05,0.3)

# elliptical aperture
# E_in = E_0 * ellipse(Xin,Yin,0.05,0.2) 


# double aperture
#E_in = E_0 * ( ellipse(Xin,Yin,0.05,0.2) + ellipse(Xin-0.25,Yin,0.05,0.2) ) 

# hole, no lens
# E_in = E_0 * ellipse(Xin,Yin,2,2)


# ------------- Step 1: separate into plane waves --------------

E_in_k = fft.fft2(E_in)         # 2D fourier transform
E_in_k = fft.fftshift(E_in_k)

# ------------- Step 2: propagate each plane wave --------------
E_out_k = E_in_k * exp(1j*k*d) * exp(-1j*(k_X**2 + k_Y**2)*d/(2*k))

# ------------- Step 2: add up the propagated plane waves --------------
E_out = fft.ifft2(E_out_k)      # inverse 2D fourier transform


## plot image
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
ax.set_title("input")
ax2.set_title("output")

ax.imshow(np.abs(E_in)**2,interpolation='bilinear',
            extent=[x.min(), x.max(), y.min(), y.max()])
            #cmap=cm.gray)
ax2.imshow(np.abs(E_out)**2,interpolation='bilinear',
            extent=[x.min(), x.max(), y.min(), y.max()])
            #cmap=cm.gray)

ax.set_xlabel("$x_\mathrm{in}$")
ax.set_ylabel("$y_\mathrm{in}$")

ax2.set_xlabel("$x_\mathrm{out}$")
ax2.set_ylabel("$y_\mathrm{out}$")

plt.show()
