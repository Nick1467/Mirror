import matplotlib.pyplot as plt
import numpy      as np
import math
#from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit

pi = math.pi

class subsystem:

    def __init__(self, name , damp):
        self.name = name
        self.damp = damp

#coupling strength
Num = 1
g_0 = 2*pi*9.8*10**6
g = 2*pi*24*10**6 #(MHz)
g_c = 0
g_d = g_0

a = subsystem('magnon', 2*pi*0.8*10**6)
b = subsystem('pump induce magnon', 2*pi*0.4*10**6)
c = subsystem('cavity photon', 2*pi*2.33*10**6)

omega   = 2 *  math.pi * np.linspace(-0.2*10**9, 0.2*10**9, 401)
omega_m = 2 *  math.pi * np.linspace(-0.2*10**9, 0.2*10**9, 301)
omega_c = 0 - 1j       * c .damp
omega_d = 0 - 1j       * b .damp

# Create a meshgrid for omega and omega_k
omega, omega_m = np.meshgrid(omega, omega_m)

# Define the S21 function initially as a complex number
def S21_value(omega_group, omega_d , g , g_d , g_c ):
    omega, omega_m = omega_group
    S21 = - c.damp/(1j*(omega - omega_c)+1j*((omega - omega_d)*g**2 + (omega - omega_m)*g_c**2 + 2*g*g_c*g_d)/(g_d**2 - (omega - omega_d)*(omega - omega_m)))
    return (S21)

S21 = S21_value((omega , omega_m) , omega_d , g , g_d , g_c )

# Flatten the data for fitting
omega_flat    = omega.ravel()
omega_m_flat  = omega_m.ravel()
S21_flat_real = np.real(S21).ravel() # 提取實部
S21_flat_imag = np.imag(S21).ravel() # 提取虛部

# use curve_fit to fitting data in real part and image part simultaneously
def S21_value_real_imag(omega_group, omega_d, g, g_d, g_c):
    omega, omega_m = omega_group
    S21 = -c.damp / (
        1j * (omega - omega_c) +
        1j * ((omega - omega_d) * g**2 + (omega - omega_m) * g_c**2 + 2 * g * g_c * g_d) /
        (g_d**2 - (omega - omega_d) * (omega - omega_m))
    )
    '''
    最重要的部分
    # 分別返回實部和虛部!!
    '''
    return np.hstack([np.real(S21).ravel(), np.imag(S21).ravel()])

# 使用 curve_fit 進行擬合
popt, pcov = curve_fit(
    S21_value_real_imag,
    (omega_flat, omega_m_flat),
    np.hstack([S21_flat_real, S21_flat_imag]),  # 拼接實部與虛部作為目標數據
    p0=[0, g, g_d, g_c]
)



omega1 = 2 * math.pi * np.linspace(-0.2*10**9, 0.2*10**9, 1001)
omega_m1 = 2 * math.pi * np.linspace(-0.2*10**9, 0.2*10**9, 1001)
omega1, omega_m1 = np.meshgrid(omega1, omega_m1) # Create meshgrid for omega1 and omega_m1
S21_fit = S21_value((omega1, omega_m1), *popt)

#plot fitting and orignal
#plot fitting and orignal
print(f'treu: {omega_d:.2e}, {g:.2e}, {g_d:.2e}, {g_c:.2e}') 
print(f'fitting: {popt}')
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(np.abs(S21), extent=[omega.min()/10**9, omega.max()/10**9, omega_m.min()/10**9, omega_m.max()/10**9],vmin=0, vmax=1, origin='lower', cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('$\Delta\omega_c$(GHz)')
plt.ylabel('$\Delta\omega_m$(GHz)')
plt.title('Origin Data of $|S_{21}|$')

plt.subplot(1, 2, 2)
plt.imshow(np.abs(S21_value((omega1, omega_m1), *popt)), extent=[omega1.min()/10**9, omega1.max()/10**9, omega_m1.min()/10**9, omega_m1.max()/10**9],vmin=0, vmax=1, origin='lower', cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('$\Delta\omega_c$(GHz)')
plt.ylabel('$\Delta\omega_m$(GHz)')
plt.title('Fitted curve for $|S_{21}|$')

plt.tight_layout()
plt.show()