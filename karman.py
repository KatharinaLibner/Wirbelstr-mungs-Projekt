import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from scipy.interpolate import griddata
import time
import matplotlib
#from Wirbelberechnung import *

#___________________________________________Funktionen_____________________________________________________
class DGL():
    
    def __init__(self,Re,n,m,h,xi,theta,omega,tspan,LU) -> None:
        self.omega = omega
        self.Psi = None
        self.Re = Re
        self.xi = xi
        self.theta = theta
        self.m = m
        self.n = n
        self.h = h
        self.tspan = tspan
        self.dw_dt = None
        self.LU = LU
    
    def Poisson(self):
        #Wirbelstärke
        omega_tilde = np.zeros((self.m,self.n))
        omega_tilde[1:-1,1:-1] = self.h**2 * np.tile(np.exp(2*self.xi[1:-1]), (self.m-2,1))*self.omega[1:-1,1:-1]     #Poisson
        omega_tilde[1:-1,-1] = np.exp(self.xi[-1])*np.sin(self.theta[1:-1])                                           #Freie Strömung
        omega_tilde= np.reshape(omega_tilde,(self.m*self.n,1))

        #Psi lösen
        self.Psi=self.LU.solve(omega_tilde)
        
    def RB_Omega(self):
    
        self.omega=self.omega.flatten()
  
        #R.B Freie Stömung
        self.omega[np.arange(self.n-1,self.m*self.n,self.n)] = 0

        #R.B Zylinder                                                
        self.omega[np.arange(0,self.n*self.m,self.n)] = (
            (self.Psi[np.arange(2,self.n*(self.m-1)+3,self.n)]-8*self.Psi[np.arange(1,self.n*(self.m-1)+2,self.n)])/(self.h**2)).flatten()

        #Ghost points
        self.omega[np.arange(0,self.n)] = self.omega[np.arange((self.m-2)*self.n,(self.m-1)*self.n)]
        self.omega[np.arange((self.m-1)*self.n,self.n*self.m)] = self.omega[np.arange(self.n,2*self.n)]
        
        self.omega = np.reshape(self.omega,(self.m,self.n))

    def omega_eq(self,i,k):
        
        #Psi berechen (Omega muss Matrix sein für Poisson)
        self.Poisson()

        #Omega RB
        self.RB_Omega()
        
        #dw_dt berechnen
        self.omega=self.omega.flatten()
        self.dw_dt = (2*np.exp(-2*self.xi[i].flatten())/
              (self.Re*self.h**2)*(
                  self.omega[k+1].flatten()+self.omega[k-1].flatten()+
                  self.omega[k+self.n].flatten()+self.omega[k-self.n].flatten()-4*self.omega[k].flatten()) +
              np.exp(-2*self.xi[i].flatten())/
              (4*self.h**2)*(
                  (self.Psi[k+1].flatten()-self.Psi[k-1].flatten())*(self.omega[k+self.n].flatten()-self.omega[k-self.n].flatten())-
                  (self.Psi[k+self.n].flatten()-self.Psi[k-self.n].flatten())*(self.omega[k+1].flatten()-self.omega[k-1].flatten())
              ))
  
    def wirbelstärke(self):
                
        #Nur Zentrale Punkte betrachten
        i = np.tile(np.arange(1,self.n-1),self.m-2)
        k = np.reshape(np.arange(0,self.m*self.n),(self.m,self.n))
        k = k[1:-1,1:-1].flatten()
        
        #y0 festlegen sodass auf dem Rand von Omega nur nullen sind
        y0 = self.omega[1:-1,1:-1].flatten()
        
        #dw_dt berechnen
        self.omega_eq(i,k)
        
        #Omega nach der Zeit berechnen
        sol = solve_ivp(lambda t, y: self.dw_dt, self.tspan, y0)
        self.omega=sol.y[:,sol.y.shape[1]-1]
        
        #Omega expandieren (Alle Werte auf dem Rand sind Null)
        temp = np.zeros((self.m,self.n))
        temp[1:-1,1:-1] = np.reshape(self.omega,(self.m-2,self.n-2))
        self.omega = temp
        del temp

    def run(self):
        self.wirbelstärke()
        self.RB_Omega()
        return self.omega
        
Re = 100        #Reynolds-Zahl

#Zeitparameter
t_start = 0
t_end = 0.005
tspan = np.array([t_start, t_end])

#Gridpunkte
n = 101
m = 202

#Gridintervalle
N = n - 1
M = m - 2       #2 ghost points

h = 2*np.pi/M   #Gridschrittweite in Bezug auf M

#Wirbelstärke
omega = np.zeros((m,n))

#xi und theta in grid
xi = np.arange(0,n)*h           
theta = np.arange(-1,(M+1))*h

#Matrix A für psi
A = np.zeros((m*n,m*n))
A[np.arange(0, m*n,n), np.arange(0,m*n,n)] = 1           #Zylinderbedingung
A[np.arange(n-1, m*n,n), np.arange(n-1, m*n, n)] = 1     #Randwerte
A[np.arange(0, n), np.arange(0, n)] = 1                  #Mittelinie hinten
A[np.arange(0, n), np.arange((m-2)*n, (m-1)*n)]  = -1    #Mittelinie hinten
A[np.arange(m*n-n, m*n), np.arange(m*n-n, m*n)]  = 1     #Mittelinie vorne
A[np.arange(m*n-n, m*n), np.arange(n, 2*n)]      = -1    #Mittelinie vorne

#Poisson-Gleichung mit FDM für A (alle nicht-RB)
for j in range(1,m-1):
    A[np.arange((j*n+1),(n-1)+j*n), np.arange((j*n+1),(n-1)+j*n)] = 4
    A[np.arange((j*n+1),(n-1)+j*n), np.arange((j*n+1),(n-1)+j*n)+1 ] = -1
    A[np.arange((j*n+1),(n-1)+j*n), np.arange((j*n+1),(n-1)+j*n)-1 ] = -1
    A[np.arange((j*n+1),(n-1)+j*n), np.arange((j*n+1),(n-1)+j*n)+n ] = -1
    A[np.arange((j*n+1),(n-1)+j*n), np.arange((j*n+1),(n-1)+j*n)-n ] = -1

#LU-Zerlegung
A=csc_matrix(A)
LU = splu(A)

#______________________________________________Funktionen_______________________________________________________________________

def Plot_Re(omega, xi, theta,i,t_end,Re, ax):
    ax.clear()
    
    #Xi-Theta grid
    xi, theta = np.meshgrid(xi,theta)
    
    #xy grid
    nx = 640
    ny = 480
    xmin = -1.5
    xmax = 21
    ymax = ((xmax-xmin)/2)*ny/nx
    ymin = -ymax
    x = np.linspace(xmin,xmax,nx+1)
    y = np.linspace(ymin,ymax,ny+1)
    x,y = np.meshgrid(x,y)
    
    #Interpolationspunkte
    xi_i = 0.5*np.log(x**2 + y**2)
    theta_i = np.mod(np.arctan2(y,x), 2*np.pi)
    points = np.array([xi.ravel(), theta.ravel()]).T
    values = omega.ravel()
    xi_theta_i = np.array([xi_i.ravel(), theta_i.ravel()]).T
    omega_xy = griddata(points, values, xi_theta_i, method='cubic')
    omega_xy = omega_xy.reshape(xi_i.shape)
    omega_xy[xi_i<0] = 0
    
    #Colormap
    levels = np.linspace(-1, 1, 1000)
    vmin, vmax = levels[0], levels[-1]
    cmap = plt.get_cmap('RdBu_r', len(levels))
    
    #Plot
    ax.imshow(omega_xy, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax, extent=[xmin, xmax, ymin, ymax])
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')
    ax.axis([xmin, xmax, ymin, ymax])
    ax.axis('off')
    
    #Text
    ax.set_title(f'Kármen Vortex Street, t = {i * t_end:.2f}', fontsize=12)
    ax.annotate(f'Re: {Re}', xy=(0.5, 0.05), xycoords='figure fraction', ha='center', fontsize=10)
    
    # Add black circle
    circle = plt.Circle((0, 0), 1, color='black',fill=False, linewidth=2)
    ax.add_artist(circle)

    #Schön plotten
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_facecolor('white')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.axis('off')
             
    return ax

#______________________________________________Hauptprogramm/Visualisierung______________________________________________________ 
dgl = DGL(Re,n,m,h,xi,theta,omega,tspan,LU)
Iterationen = 25000
fig, ax = plt.subplots()
t_0=time.time()
for I in range(100000):     #Vorlauf
    omega=dgl.run()
t_1=time.time()
print('Zeit für Iterationen Vorlauf:',(t_1-t_0)/60)
t_0=time.time()
def update(i):
    omega = dgl.run()
    Plot_Re(omega, xi, theta,i,t_end,Re, ax)
    return []  

ani = animation.FuncAnimation(fig, update, frames=Iterationen, interval=48, blit=True, repeat=Iterationen)
writer = animation.FFMpegWriter(fps=200,bitrate=1800)   #Video gestalter
ani.save(f'KVS_Re={Re}.mp4', writer)
t_1=time.time()
print('Zeit für Visualisierung:',(t_1-t_0)/60)
