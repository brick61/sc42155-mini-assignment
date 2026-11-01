import numpy as np
import matplotlib.pyplot as plt

#Q1.6 printing a closed path of filament
#closed path split into N cells C_0 ... C_N-1, cross-sections S_0 ... S_N with S_N = S_0
#energy balance per cell: dU_i/dt = Phi[S_i] - Phi[S_i+1] + Q_i
#cell i is printed at t_i = i*ds/v with temperature T_n (growing domain)

#PLA filament
ro = 1250.0 #kg/m^3
c = 1800.0 #J/kgK
k = 0.13 #W/mK
a = 0.4*10**-3 #m, side of square cross-section

#heat transfer
h_e = 10.0 #W/m^2K, convection to air
h_b = 500.0 #W/m^2K, contact with bed
T_n = 200.0 #celsius, filament leaving the nozzle
T_e = 20.0 #celsius, air
T_b = 60.0 #celsius, bed

#path
A = 50*10**-3 #m, semi-axis ellipse
B = 30*10**-3 #m, semi-axis ellipse
v = 50*10**-3 #m/s, print speed
L = np.pi*(3*(A+B) - np.sqrt((3*A+B)*(A+3*B))) #m, perimeter ellipse (Ramanujan)
t_p = L/v #s, print time

#cells
N = 1276
ds = L/N #m
V = a**2*ds #m^3
C_cell = ro*c*V #J/K, heat capacity of one cell
G_k = k*a**2/ds #W/K, conduction between two cells
G_e = h_e*3*a*ds #W/K, cell to air
G_b = h_b*a*ds #W/K, cell to bed

alpha = k/(ro*c) #m^2/s
phi_e = 3*h_e/(ro*c*a) #1/s
phi_b = h_b/(ro*c*a) #1/s
phi = phi_e + phi_b #1/s
T_inf = (phi_e*T_e + phi_b*T_b)/phi #celsius
Bi = h_b * a /k
eps = 1.0 #K, tolerance uniform temperature

#time
dt = ds/v #s, one new cell per time step
dt_max = 1/(2*alpha/ds**2 + phi) #s, stability forward Euler
if dt > dt_max:
    print('WARNING: time step too large, dt =', dt, 's > dt_max =', dt_max, 's')

t_0 = 0 #s
t_1 = 20 #s
t = np.arange(t_0, t_1+dt, dt)

def flows(T, m, closed):
    "Input:"
    "T: temperature of the cells (celsius), array of N floats"
    "m: number of printed cells, int"
    "closed: True when the path is closed, bool"
    "Output:"
    "Phi: heat flow through cross-sections S_0 ... S_N (W), array of N+1 floats"
    Phi = np.zeros(N+1)
    Phi[1:m] = G_k*(T[0:m-1] - T[1:m]) #between printed cells
    if closed:
        Phi[N] = G_k*(T[N-1] - T[0]) #last cell touches the first cell
        Phi[0] = Phi[N]
    return Phi

def sources(T, m):
    "Input:"
    "T: temperature of the cells (celsius), array of N floats"
    "m: number of printed cells, int"
    "Output:"
    "Q: heat to the cells from air and bed (W), array of N floats"
    Q = np.zeros(N)
    Q[0:m] = G_e*(T_e - T[0:m]) + G_b*(T_b - T[0:m])
    return Q

def numeriek(t):
    T_num = np.full((N, len(t)), np.nan) #space along rows, time along columns, nan = not printed
    T = np.zeros(N)
    T[0] = T_n #first cell leaves the nozzle at t = 0
    m = 1
    T_num[0:m, 0] = T[0:m]
    for i in range(len(t)-1):
        closed = (m == N)
        Phi = flows(T, m, closed)
        Q = sources(T, m)
        T[0:m] = T[0:m] + dt*(Phi[0:m] - Phi[1:m+1] + Q[0:m])/C_cell
        if m < N:
            T[m] = T_n #nozzle deposits the next cell
            m = m + 1
        T_num[0:m, i+1] = T[0:m]
    return T_num
T_num = numeriek(t)

def T_lumped(s, t):
    "Input:"
    "s: position along the path (m), float"
    "t: time (s), float"
    "Output:"
    "T: temperature without conduction along the filament (celsius), float"
    T = T_inf + (T_n - T_inf)*np.exp(-phi*(t - s/v))
    return T

#cooling times
T_max = np.nanmax(T_num, axis=0)
T_min = np.nanmin(T_num, axis=0)
t_b = t[-1]
t_u = t[-1]
for i in range(len(t)):
    if not np.isnan(T_num[N-1, i]) and T_max[i] <= T_b:
        t_b = t[i]
        break
for i in range(len(t)):
    if not np.isnan(T_num[N-1, i]) and T_max[i] - T_min[i] < eps:
        t_u = t[i]
        break

t_b_an = t_p + np.log((T_n - T_inf)/(T_b - T_inf))/phi
t_u_an = t_p + np.log((T_n - T_inf)*(1 - np.exp(-phi*t_p))/eps)/phi

#lumped elements: coldest is the first cell, hottest the last printed cell
s_last = (N-1)*ds #m
T_max_an = np.zeros(len(t))
T_min_an = np.zeros(len(t))
for i in range(len(t)):
    T_min_an[i] = T_lumped(0, t[i])
    if t[i] < s_last/v:
        T_max_an[i] = T_n
    else:
        T_max_an[i] = T_lumped(s_last, t[i])

i_p = int(round(t_p/dt))
print('Biot number =', round(Bi, 3))
print('alpha =', alpha, 'm^2/s')
print('phi =', round(phi, 4), '1/s')
print('T_inf =', round(T_inf, 2), 'celsius')
print('path length =', round(L*1000, 1), 'mm')
print('print time =', round(t_p, 2), 's')
print('dt =', round(dt*1000, 2), 'ms, dt_max =', round(dt_max, 3), 's')
print('T range at t_p =', round(np.nanmin(T_num[:, i_p]), 1), '-', round(np.nanmax(T_num[:, i_p]), 1), 'celsius')
print('t_b numerical =', round(t_b, 2), 's, lumped =', round(t_b_an, 2), 's')
print('t_u numerical =', round(t_u, 2), 's, lumped =', round(t_u_an, 2), 's')

#temperature along the filament
s = (np.arange(N) + 0.5)*ds #m, cell centres
t_plot = [1.0, 2.5, t_p, 7.0, 10.0, t_b]

fig_single, ax_1 = plt.subplots(figsize=(5, 4))
for tt in t_plot:
    i_t = int(round(tt/dt))
    ax_1.plot(s*1000, T_num[:, i_t], label='t = '+str(round(tt, 1))+' s')
ax_1.axhline(T_b, color='black', linestyle='--', label='T_b')
ax_1.set_xlabel("position along path s (mm)")
ax_1.set_ylabel("temperature (celsius)")
ax_1.grid()
ax_1.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig_single.savefig('T wrt s.pdf', dpi=300, bbox_inches='tight')

#hottest and coldest cell
fig_single, ax_2 = plt.subplots(figsize=(5, 4))
ax_2.plot(t, T_max, label='max T (numerical)')
ax_2.plot(t, T_min, label='min T (numerical)')
ax_2.plot(t, T_max_an, color='black', linestyle=':', label='lumped, no conduction')
ax_2.plot(t, T_min_an, color='black', linestyle=':')
ax_2.axhline(T_b, color='black', linestyle='--', label='T_b')
ax_2.axvline(t_b, color='red', linestyle='--', label='t_b')
ax_2.set_xlabel("time (s)")
ax_2.set_ylabel("temperature (celsius)")
ax_2.grid()
ax_2.legend(loc='center right')
fig_single.savefig('T wrt t.pdf', dpi=300, bbox_inches='tight')

plt.show()
