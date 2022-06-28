#!/usr/bin/env python3
####################################################
# SCRIPT TO SHOW THE STRUCTURES FROM 3D Spectral cubes
#
#####################################################
# V 2.0 We create a script to create a movie   10/12/2019
# V 2.1: Now the movies are created in their directory   11/12/2019
# V 2.2; Now it deletes pngs and creates movies with name
#        of the source and RMS
# V 3.0: Mayor problem solved
#       cube3.spatial_coordinate_map[1][:] is the X not the Y
# V 3.1: 13/12/19. We set the coordinates wrt the center of phases.
#        Now the plot of the star makes sense
# V 3.2: NOw we can plot fluxes
# V.3.3 : Add an option for the transparecy of the points
# V.3.4 : We now can add the size in X and Y and if the vel field is switch to spatial, too
# v.3.5: Now we center the maps in the source
# v.3.6: We added Zorder to avoid overlapping problems of the scatter points.
# v.3.7: We read New ALMA data and solve the problem with NaNs
# v.4.0: Fixed for new Astropy versions
# v.4.1: Header key 'RESTFREQ' is not called this way in gildas fits. It's RESTFRQ. We added a if to solve it.
# v.4.2: Vel axis can be in km/s
# TBD: create a folder so the images are stored in #source#/#files#
#      Fix the script so the files are in the directory as the videos created
#
#  TBD : Choose the marker size depending on the point density (y>1).sum()
####################################################


import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm

import astropy.units as u
# from astropy.utils.data import download_file
from astropy.io import fits  # We use fits to open the actual data file

from astropy.utils import data
from matplotlib import colors

data.conf.remote_timeout = 60

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

from spectral_cube import SpectralCube

from scipy.stats import norm
from math import pi


from astroquery.esasky import ESASky
from astroquery.utils import TableList
from astropy.wcs import WCS
from reproject import reproject_interp

import subprocess
import os, sys

import pyvista as pv

##################################################################
# COLORS
##################################################################
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# sample
# print(f"{bcolors.WARNING}Warning: No active frommets remain. Continue?{bcolors.ENDC}")

os.system("clear")

print("---------------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------------")
print(f"{bcolors.BOLD}{bcolors.FAIL} _______  _______ _________ _______           _______ _________          _______  _______  _______ {bcolors.ENDC}")
print(f"{bcolors.BOLD}{bcolors.FAIL}(  ___  )(  ____ \\\__   __/(  ____ )|\     /|(  ____ \\\__   __/|\     /|(  ____ )(  ____ \(  ____ \{bcolors.ENDC}")
print(f"{bcolors.BOLD}{bcolors.FAIL}| (   ) || (    \/   ) (   | (    )|| )   ( || (    \/   ) (   | )   ( || (    )|| (    \/| (    \/{bcolors.ENDC}")
print(f"{bcolors.BOLD}{bcolors.FAIL}| (___) || (_____    | |   | (____)|| |   | || |         | |   | |   | || (____)|| (__    | (_____ {bcolors.ENDC}")
print(f"{bcolors.BOLD}{bcolors.FAIL}|  ___  |(_____  )   | |   |     __)| |   | || |         | |   | |   | ||     __)|  __)   (_____  ){bcolors.ENDC}")
print(f"{bcolors.BOLD}{bcolors.FAIL}| (   ) |      ) |   | |   | (\ (   | |   | || |         | |   | |   | || (\ (   | (            ) |{bcolors.ENDC}")
print(f"{bcolors.BOLD}{bcolors.FAIL}| )   ( |/\____) |   | |   | ) \ \__| (___) || (____/\   | |   | (___) || ) \ \__| (____/\/\____) |{bcolors.ENDC}")
print(f"{bcolors.BOLD}{bcolors.FAIL}|/     \|\_______)   )_(   |/   \__/(_______)(_______/   )_(   (_______)|/   \__/(_______/\_______){bcolors.ENDC}")
print("")
print("---------------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------------")
print("")
print(f"                                        {bcolors.BOLD}{bcolors.UNDERLINE}ASTRUCTURES V4.2{bcolors.ENDC}")
print(f"{bcolors.BOLD}{bcolors.OKBLUE}                                Author: Guillermo Quintana-Lacaci  {bcolors.ENDC}")
print(f"{bcolors.BOLD}                                           11-03-2022  {bcolors.ENDC}")
print(f"{bcolors.BOLD}{bcolors.HEADER}                                      guillermo.q@csic.es  {bcolors.ENDC}")
print("")
print("---------------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------------")


########################################################################






#import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17})

##################################################################
# INPUTS FROM THE USER
##################################################################
import optparse

parser = optparse.OptionParser()
parser.add_option("-f", "--fname", dest="fname", help="Name of the fits file w/o extension")
parser.add_option("-t", "--threshold", dest="rms", help="Flux threshold  = X times sigma (Jy/beam")
parser.add_option("-v", "--velocity", dest="vfield", help="Velocity field: 0 = Z as velocity, 1 = constant, 2 = radial")
parser.add_option("-c", "--color", dest="color", help="Layout of the plot: White (w) or black (b) without axis")
parser.add_option("-a", "--axis", dest="a2p", help="Select the plots. X or Z: rotation around x,z. V: Vel splices")
parser.add_option("-n", "--noise", dest="rmstype", action="store_true", help="If the source is extended the rms determination might not be accurate in the intermediate channels. Best to use the first channel without source. Type this option if this is the case")
parser.add_option("-k", "--keep", dest="keepfig", action="store_true", help="Type -k to keep the images. Otherwise they will be deleted after creating the videos.")
parser.add_option("-F", "--flux", dest="plotflux", action="store_true", help="If selected, the plots will show the flux scale for each point. If not all points have the same flux = vel. The latter is recommended to see weak structures")
parser.add_option("-o", "--opacity", dest="opacity", help="Opacity of the points. L(ow), M(edium) or H(high) ")
parser.add_option("-s", "--size", dest="size", help="Size of the cube in arcseconds")
parser.add_option("-x", "--center", dest="center", nargs=3, help="Center of the maps", type="float")
# parser.add_option("-vf", "--vfield", dest="vfield2", nargs=3, help="Ad hoc velocity field: vo, ro, alpha", type="float")

if len(sys.argv) == 1:
        parser.parse_args(['--help'])
        exit()

(options, arguments) = parser.parse_args()

fname2 = str(options.fname) + ".fits"
fname = str(options.fname)
rms = float(options.rms)
vfield = options.vfield
# vfield2 = options.vfield2
fondo = options.color
a2p = str(options.a2p)
rmstype = options.rmstype
keepfig = options.keepfig
#print(keepfig)
plotflux = options.plotflux
opacity = options.opacity
size = options.size
center = options.center


center = np.asarray(center)
X_center = center[0] * u.arcsec
Y_center = center[1] * u.arcsec
Z_center = center[2] * u.kilometer/u.second
# vfield2 = np.asarray(vfield2)
# print(center[0]*u.arcsec)
#print(options.size)

# print(type(rms))
# exit()


dir1 = fname
haydir = os.path.isdir(dir1)
if not haydir:
    subprocess.run(["mkdir", dir1])

if opacity == 'H':
    alphav = 1
elif opacity == 'M':
    alphav = 0.3
else:
    alphav = 0.01


##################################################################
# START READING THE CUBE + FILTER PIXELS BELOW NOISE
##################################################################

data = fits.open(fname2)  # Open the FITS file for reading
hdr = data[0].header
if data[0].header['CTYPE3'] == 'FREQUENCY': # we check the units of the axis and correct its name
    data[0].header.set('CTYPE3','FREQ')
#if data[0].header['BUNIT'] == 'K (Ta*)': # we check the units of the axis and correct its name
    #data[0].header.set('BUNIT','Jy/beam')
#cube = SpectralCube.read(data)  # Initiate a SpectralCube
cube = SpectralCube.read(fname2).with_fill_value(0)  #just in case we have NANs
data.close()  # Close the FITS file - we already read it in and don't need it anymore!


print("---------------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------------")
print(f"{bcolors.OKGREEN}[+] SOURCE: {bcolors.ENDC}" + hdr['object'])
print(f"{bcolors.OKGREEN}[+] Fits file: {bcolors.ENDC}" + fname2)
print("---------------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------------")
print(cube)

#################################################################
# WE ESTIMATE THE NOISE
#################################################################
print("---------------------------------------------------------------------------------------------------")
print(f"{bcolors.OKGREEN}[+] ESTIMATE THE GAUSSIAN NOISE{bcolors.ENDC}")
sigmas = []
mus = []


if rmstype:
    #print("****************************************")
    print(f"{bcolors.OKGREEN}       [+] Obtaining the RMS from the first channel{bcolors.ENDC}")
    (mu, sigma) = norm.fit(cube[0])
    sigma2 = sigma

    sigma = sigma*rms
    if hdr['bunit'] == 'Jy/beam':
        units = 'mJy/beam'
        print(f"{bcolors.OKGREEN}       [+] RMS = {bcolors.ENDC}" + str(sigma2*1000) + " mJy/beam")
    else:
        units = 'bunit'
        print(f"{bcolors.OKGREEN}       [+] RMS = {bcolors.ENDC}" + str(sigma2) + str(hdr['bunit']))
    #sigma = 0.16435944

else:
    #print("****************************************")
    print(f"{bcolors.OKGREEN}[+] Obtaining the RMS as the mean RMS of all channels{bcolors.ENDC}")

    for vel in range(len(cube)):
        (mu, sigma) = norm.fit(cube[vel])
        # print(mu,sigma)
        sigmas = np.append(sigmas, sigma)
        mus = np.append(mus, mu)

    mu = mus.mean()
    sig_ratio = abs(mu / mus[0])
#    print(mu, mus[0])

    if sig_ratio > 100:
        print(f"{bcolors.WARNING}[-] WARNING: THE MEAN ZERO OF THE GAUSSIAN DISTRIBUTION IS SIGNIFICANTLY HIGHER EXPECTED {bcolors.ENDC}")
        print(f"{bcolors.WARNING}     [-] RMS IS PROBABLY DOMINATED BY THE SOURCE {bcolors.ENDC}")
        print(f"{bcolors.WARNING}     [-] IS YOUR SOURCE TOO EXTENDED? {bcolors.ENDC}")
        print(f"{bcolors.WARNING}     [-] TRY THE -n OPTION. {bcolors.ENDC}")
        print(f"{bcolors.FAIL}[-] EXITING. {bcolors.ENDC}")
        exit()

    sigma2 = sigmas.mean()
    mu = mus.mean()

    #print(sigma, mu)
    sigma = sigma2*rms
    if hdr['bunit'] == 'Jy/beam':
        units = 'mJy/beam'
        print(f"{bcolors.OKGREEN}       [+] RMS = {bcolors.ENDC}" + str(sigma2*1000) + " mJy/beam")
    else:
        print(f"{bcolors.OKGREEN}       [+] RMS = {bcolors.ENDC}" + str(sigma2) + str(hdr['bunit']))
    #sigma = 0.16435944

##################################################################
# WE MASK THE DATA BELLOW THE NOISE
#################################################################
print("---------------------------------------------------------------------------------------------------")
print(f"{bcolors.OKGREEN}[+] MASKING DATA BELOW THE NOISE{bcolors.ENDC}")
if hdr['bunit'] == 'Jy/beam':
    mask = cube > sigma*u.Jy/u.beam
if hdr['bunit'] == 'K':
    mask = cube > sigma*u.K
#mask = cube > rms*u.Jy/u.beam
cube2 = cube.with_mask(mask)
cube3 = cube2.with_fill_value(0.)

#cube[10, :, :].quicklook()

#for vel in range(len(cube3)):
#    print(cube[vel])
#    print(cube2[vel].max())
#    print(cube3[vel].max())

#############################################
# IF SPECTRAL AXIS IS IN FREQ WE CONVERT IT TO VEL
if hdr['CTYPE3'] == 'FREQ':
    print(f"{bcolors.OKBLUE}[+] CHANGING SPECTRAL AXIS TO VELOCITY{bcolors.ENDC}")
    if 'RESTFREQ' in hdr:
	    restfreq = hdr['RESTFREQ'] * u.Hz  # rest frequency from the header
	    freq_to_vel = u.doppler_radio(restfreq)
	    vel_axis = cube.spectral_axis.to(u.km / u.s, equivalencies=freq_to_vel)
    elif 'RESTFRQ' in hdr:
	    print("Key RESTFRQ")
	    restfreq = hdr['RESTFRQ'] * u.Hz  # rest frequency from the header
	    freq_to_vel = u.doppler_radio(restfreq)
	    vel_axis = cube.spectral_axis.to(u.km / u.s, equivalencies=freq_to_vel)
else:
    vel_axis = cube.spectral_axis
#############################################



#3D PIXEL SIZE IN ARCSEC X ARCSEC X KM/S
mark_size = [abs(cube.header['CDELT1'])*60*60,abs(cube.header['CDELT2'])*60*60,abs(cube.header['CDELT3']/1000)]
if mark_size[2] > 2:
    mark_size = 1.2

#mark_size = 20

# mark_size = 100*mark_size


# cube3.spectral_axis.to(u.km/u.s)
# cube3.spatial_coordinate_map   # el [0] es la distacia relativa y el [1] la absoluta.

#################################################################
# We obtain the mean values for the plots and limits
##################################################################
print("---------------------------------------------------------------------------------------------------")
print(f"{bcolors.OKGREEN}[+] SETTING LIMITS OF THE PLOTS{bcolors.ENDC}")
# zmin = cube3.spectral_axis.min()
zmin = vel_axis.min()
zmax = vel_axis.max()
# zmax = cube3.spectral_axis.max()
if zmin == zmax and zmax == 0:
    print("---------------------------------------------------------------------------------------------------")
    print(f"{bcolors.WARNING}     [-] NO POINTS LEFT AFTER MASKING{bcolors.ENDC}")
    print(f"{bcolors.WARNING}     [-] TRY TO REDUCE THE THRESHOLD{bcolors.ENDC}")
    print(f"{bcolors.FAIL}[-] EXITING {bcolors.ENDC}")
    print("---------------------------------------------------------------------------------------------------")
    exit()
##################################################################
#x_mean = np.mean(cube3.spatial_coordinate_map[1][:])
#y_mean = np.mean(cube3.spatial_coordinate_map[0][:])
##################################################################
print("ERROR TEST 0")
x = cube3.spatial_coordinate_map[1][:]
print("ERROR TEST 1")
# x = x-x_mean
if 'RA' in hdr:
    x = x-hdr['RA']*u.deg
elif 'OBSRA'   in hdr:
    x = x - hdr['OBSRA']*u.deg

x = x.to(u.arcsec)
xmin = x.min()
xmax = x.max()
##################################################################
y = cube3.spatial_coordinate_map[0][:]\
# y = y-y_mean
if 'DEC' in hdr:
    y = y-hdr['DEC']*u.deg
elif 'OBSDEC'   in hdr:
    y = y-hdr['OBSDEC']*u.deg
y = y.to(u.arcsec)
ymin = y.min()
ymax = y.max()

###################################################################
# INITIALIZE THE ARRAYS WHICH WOULD STORE OUR 3D coordinates & the star
####################################################################
if x.unit == "arcsec":
	x2 = [] * u.arcsec
	print( "X in arcsec")
if y.unit == "arcsec":
	y2 = [] * u.arcsec
	print( "Y in arcsec")
if vel_axis[0].unit == 'm / s':
	z2 = [] * u.meter/u.second
	print( "Z in m/s")
if vel_axis[0].unit == 'km / s':
	z2 = [] * u.meter/u.second
	print( "Z in km/s")
if cube3[0].unit == 'Jy / beam':
	flux2 = [] * u.Jansky/u.beam
	print( "Flux in Jy/beam")

#xstar, ystar, zstar = -7.754854115081302735E-02 , 0.162628630074047598, 22
xstar, ystar, zstar = 0.0, 0.0, center[2]
zstar2 = zstar
####################################################################
#NOW WE CREATE A CUBE TO REPLACE THE FLUX VALUES WITH THE VELOCITIES
#TO HAVE A CLOUD OF POINTS IN THE COORDINATES X,Y,V.
####################################################################

for vel in range(len(cube3)):
    #print("vel="+str(vel))
    # z= np.where(cube3[vel]>0.,cube.spectral_axis[vel],0.)
    #print(flux.mean())
    flux = cube3[vel]
    if flux.unit == 'Jy/beam':
        flux = flux*1000
    z= np.where(cube3[vel]>0.,vel_axis[vel],0.)
    # flux = np.where(cube3[vel]>0.,cube3[vel],0.)
    #z = z/1000
    z = z.to(u.kilometer/u.second)
    masks = (abs(z) > 0.)
    z1 = z[masks]
    x1 = x[masks]
    y1 = y[masks]
    flux1 = flux[masks]
    # z = cube3[vel]
    # ax.scatter(x1, y1, -z1, zdir='z', s=1, cmap='viridis')
    # ax.scatter(x1, y1, -z1, zdir='z', s=1, cmap='Greys', alpha=0.2)
                #print(index)

    z2 = np.append(z2,z1)
    y2 = np.append(y2,y1)
    x2 = np.append(x2,x1)
    flux2 = np.append(flux2,flux1)

print(x2.unit)
print(X_center.unit)

x2 = x2 - X_center #center[0]
y2 = y2 - Y_center #center[1]

#WE SET THE LIMINTS
xmax = x2.max()
xmin = x2.min()
ymax = y2.max()
ymin = y2.min()
zmax = z2.max()
zmin = z2.min()

############
# THINKING OF A WAY TO DEAL WITH the marker size
############

points = (z2 > 0. ).sum()
volume = (xmax-xmin)*(ymax-ymin) #*(zmax-zmin)
density = points/volume
#print ("#points    Volume     density")
#print(points, volume, density)

mark_size = 0.1
#mark_size = mark_size*np.sqrt(330/volume)/10*100
mark_size = mark_size*np.sqrt(330/volume)/10*100*size
print("qqqq", mark_size)
#mark_size = mark_size*np.sqrt(330/volume)/50
#print(mark_size)
#adapt the marker size to the size of the plot
#exit()



#####################################################################
# MODIFY THE VELOCITY AXIS TO OBTAIN A SPATIAL COORDINATE ASSUMING A
# CERTAIN VELOCITY FIELD
#####################################################################

#TBD


if vfield == "2":
    r0 = 0.1
    v0 = 17
    vlsr = Z_center #center[2]
    #print(z2[0])
    z3 = (z2-vlsr)*r0/v0
    #print("convert", (z2[0]-vlsr)*r0/v0)
    z3 = z3.to(u.kilometer/u.second) 
    zstar2 = (22-vlsr.value)*r0/v0
    zstar=0
    legend = str(fname) #+"; v = "+str(v0)+"r/"+str(r0)
    #legend = str(fname)+"; v = "+str(v0)+"r/"+str(r0)
    zmax = z3.max()
    zmin = z3.min()
    zlabel = "Z (\'\')"
elif vfield == "1":
    v0 = 50
    vlsr = Z_center #center[2]
    z2 = z2 - vlsr
    xy = np.sqrt((x2-xstar)**2+(y2-ystar)**2)
    ratio = z2/v0
    z3 = ratio*xy/np.cos(np.arcsin(z2/v0))
    z3 = z3.to(u.kilometer/u.second) 
    zstar=0
    legend = str(fname)+"; v = "+str(v0)+"km/s"
    zmax = z3.max()
    zmin = z3.min()
    zlabel = "Z (\'\')"
elif vfield == "0":
    legend = str(fname)
    z3 = z2
    print("otro test", z3[0])
    z3 = z3.to(u.kilometer/u.second)  
    print("otro test", z3[0])
    zmax = z3.max()
    zmin = z3.min()
    zlabel = "Vel (km/s)"
else :
    print("WRONG VELOCITY FIELD")
    exit()


#####################################################################
# COLOR SCHEME
#####################################################################
if plotflux:
    fname = fname + "_flux"

gamma_colorb = 'Wistia'
#gamma_color = 'Greys'
#gamma_color = 'seismic'
#gamma_color = 'winter'
#gamma_color = 'cool'
gamma_color = 'jet'
# gamma_color = 'hot'
####################################################################################################################################3
####################################################################################################################################3
########################################PLOTS#######################################################################################3
####################################################################################################################################3
####################################################################################################################################3
####################################################################################################################################3
####################################################################################################################################3
# We set the size if introduced in the call
if options.size != None:
    size = float(size)
    size = size*u.arcsec
    xmin = -size/2
    xmax = size/2
    ymin = -size/2
    ymax = size/2
    # if vfield != "0":
        # zmin = -size.value/2
        # zmax = size.value/2
###############################################################################


if a2p != "Z" and a2p != "V":
##################################################################3
#  WE CREATE THE FIGURE TO SIMULATE A ROTATION IN X
###################################################################
    print("---------------------------------------------------------------------------------------------------")
    print(f"{bcolors.OKGREEN}[+] ROTATING THE STRUCTURE AROUND X{bcolors.ENDC}")

    for angle in range(0,360):

        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        ##############################################
        cstar = 'black'
        if fondo == "b":
            plt.style.use('dark_background')
            # plt.axis("off")
            ax = plt.axes(projection="3d")
            cstar = 'white'
            gamma_color = gamma_colorb
            ax.grid(False)
            ax.set_frame_on(False)
            # make the panes transparent
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        else:
            ax = plt.axes(projection="3d")
        ###########################################
        ax.view_init(30,angle)
        # bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # width, height = bbox.width, bbox.height
        # print(bbox)
        # print(width)
        # exit()
        ###########################################
        if plotflux:
            img = ax.scatter3D(x2.value, y2.value, z3, c=flux2, cmap=plt.hot(), alpha=alphav, s=mark_size, zorder=z3)
            fig.colorbar(img).set_label(hdr['bunit'], rotation=90)
        else:
            ax.scatter3D(x2.value, y2.value, z3, c=z2, cmap=gamma_color, alpha=alphav, s=mark_size, zorder=z3)
        ###########################################
        ax.scatter3D(xstar, ystar, zstar, marker='*', c=cstar, cmap=gamma_color, alpha=1, s=50, zorder=z3)
        ax.scatter3D(xstar, ystar, zstar2, marker='o' , c=cstar, cmap=gamma_color, alpha=1, s=50, zorder=z3)
        ############################################
        ax.set_xlim(xmax.value, xmin.value)
        ax.set_ylim(ymin.value, ymax.value)
        ax.set_zlim(zmin, zmax)
        ax.set_ylabel('\u03B4 Dec. (\'\')')
        ax.set_xlabel('\u03B4 R.A. (\'\')')
        ax.set_zlabel(zlabel)
        ax.set_title(legend)
        # plt.savefig('test_angle'+str(angle)+'.png')
        if angle < 10:
            plt.savefig(dir1 + "/"+ fname+'_angle_00'+str(angle)+'.png')
        elif angle >= 10 and angle < 100:
            plt.savefig(dir1 + "/"+ fname+'_angle_0'+str(angle)+'.png')
        elif angle >= 100:
            plt.savefig(dir1 + "/"+ fname+'_angle_' + str(angle) + '.png')

        plt.close()

    #print("****************************************")
    print(f"{bcolors.OKGREEN}     [+] Creating the video{bcolors.ENDC}")
    movie_maker = "ffmpeg -y -loglevel panic -framerate 25 -i " + dir1 + "/" + fname + "_angle_%03d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + dir1 + "/" + str(hdr['object']) + "_" + fname + "_ANGLE_" + str(rms) +".mp4"
    os.system(movie_maker)

if a2p != "X" and a2p != "V":
##################################################################3
#  WE CREATE THE FIGURE TO SIMULATE A ROTATION IN Z
###################################################################
    print("---------------------------------------------------------------------------------------------------")
    print(f"{bcolors.OKGREEN}[+] ROTATING THE STRUCTURE AROUND Z{bcolors.ENDC}")


    for angle in range(0,360):
        fig = plt.figure()
        fig.set_size_inches(10.5, 10.5)
        cstar = 'black'
        ##############################################
        if fondo == "b":
            plt.style.use('dark_background')
            #plt.axis("off")
            ax = plt.axes(projection="3d")
            cstar = 'white'
            gamma_color = gamma_colorb
            ax.grid(False)
            ax.set_frame_on(False)
            # make the panes transparent
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        else:
            ax = plt.axes(projection="3d")
        ##########################################
        #ax.view_init(0, angle)
        ax.view_init(15, angle)
        ###########################################
        ###########################################
        if plotflux:
#            camdis =  z3*np.cos(np.radians(-angle))+x2.value*np.cos(np.radians(-angle)) # np.sqrt(y2.value*y2.value+x2.value*x2.value)*np.sin(np.radians(angle))
            #print(camdis)
 #           lowcamdis = min(camdis)
 #           camdis = camdis + abs(lowcamdis)
 #           camdis = camdis*10
#            img = ax.scatter3D(z3, x2.value, y2.value, c=flux2, cmap=gamma_color, alpha=alphav, s=mark_size, zorder=camdis
#            print("lalal", mark_size)
            img = ax.scatter3D(z3, x2.value, y2.value, c=flux2, cmap=gamma_color, alpha=alphav, s=mark_size, zorder=z3)
            fig.colorbar(img).set_label(hdr['bunit'], rotation=90)
        else:
            ax.scatter3D(z3, x2.value, y2.value, c=z2, cmap=gamma_color, alpha=alphav, s=mark_size, zorder=z3)
        ###########################################
        ax.scatter3D(zstar, xstar, ystar, marker='*' , c=cstar, cmap=gamma_color, alpha=1, s=50, zorder=z3)
        ax.scatter3D(zstar2, xstar, ystar, marker='o' , c=cstar, cmap=gamma_color, alpha=1, s=50, zorder=z3)
        ############################################
        #xline = np.array([0.8,0.14])    #RA
        #yline = np.array([-0.38,-0.06])   #DEC
        #zline = np.array([0.1,0.02])  #Z
        #ax.plot(zline, xline, yline, c='black', linewidth=3.0)
        ############################################
        #xline = np.array([0.8,-0.2])    #RA
        #yline = np.array([-0.38,0.1])   #DEC
        #zline = np.array([0.1,-0.025])  #Z
        #ax.plot(zline, xline, yline, c='red', linewidth=3.0)
        # draw size of source
        #u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        #xesf = 5.5e-3 * np.cos(u) * np.sin(v) + xstar
        #yesf = 5.5e-3 * np.sin(u) * np.sin(v) + ystar
        #zesf = 5.5e-3 * np.cos(v) + zstar
        #ax.plot_wireframe(zesf, xesf, yesf, color="r")
        ##pintamos el beam
        #u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        #xesf = 0.02/2 * np.cos(u) * np.sin(v) + xmin.value*0.8
        #yesf = 0.02/2 * np.sin(u) * np.sin(v) + ymin.value*0.8
        #zesf = 0.03/2 * np.cos(v) + zmin*0.8
        #ax.plot_wireframe(zesf, xesf, yesf, color="r")
        ############################################
        ax.set_ylim(xmax.value, xmin.value)
        ax.set_zlim(ymin.value, ymax.value)
        ax.set_xlim(zmax.value, zmin.value)
        ax.set_ylabel('\u03B4 R.A. (\'\')')
        ax.set_xlabel(zlabel)
        ax.set_zlabel('\u03B4 Dec. (\'\')')
        ax.set_title(legend)
        #plt.show()
        # plt.savefig('test_2angle'+str(angle)+'.png')
#        if angle == 0:
#            plt.show()
        if angle < 10:
            plt.savefig(dir1 + "/"+ fname+'_2angle_00'+str(angle)+'.png')
        elif angle >= 10 and angle < 100:
            plt.savefig(dir1 + "/"+ fname+'_2angle_0'+str(angle)+'.png')
        elif angle >= 100:
            plt.savefig(dir1 + "/"+ fname+'_2angle_' + str(angle) + '.png')

        plt.close()

    #print("****************************************")
    print(f"{bcolors.OKGREEN}     [+] Creating the video{bcolors.ENDC}")
    movie_maker = "ffmpeg -y -loglevel panic -framerate 25 -i " + dir1 + "/" + fname + "_2angle_%03d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + dir1 +"/" + str(hdr['object']) + "_" + fname + "_ANGLE2_" + str(rms) +".mp4"
    os.system(movie_maker)

if a2p != "X" and a2p != "Z":
##################################################################3
#  WE CREATE THE FIGURES TO SHOW THE VELOCITY CHANNELS
###################################################################
    print("---------------------------------------------------------------------------------------------------")
    print(f"{bcolors.OKGREEN}[+] OVERPLOTING VELOCITY SLICES ON THE STRUCTURE{bcolors.ENDC}")

    for vel in range(len(cube3)):
        # z= np.where(cube3[vel]>0.,cube.spectral_axis[vel],0.)
        z = np.where(cube3[vel] > 0., vel_axis[vel], 0.)
#        z = z/1000
        z = z.to(u.kilometer/u.second)
        print("TEST",z[0])
        masks = (z > 0.)
        z1 = z[masks]
        x1 = x[masks]
        y1 = y[masks]   # We Obtain the points for each channel
                        # WE PLOT THE FIGURES, FIRST THE ONE WITH ALL THE CHANNELS
        x1 = np.asarray(x1) - X_center #center[0]
        y1 = np.asarray(y1) - Y_center #center[1]
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
##############################################
        cstar = 'black'
        if fondo == "b":
            plt.style.use('dark_background')
            #plt.axis("off")
            ax = plt.axes(projection="3d")
            cstar = 'white'
            gamma_color = gamma_colorb
            ax.grid(False)
            ax.set_frame_on(False)
            # make the panes transparent
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        else:
            ax = plt.axes(projection="3d")
        ##########################################
        ax.view_init(10, 60)
        # ax.view_init(0, 90)
        ###########################################
        alphav = 0.01 # We need the points to be super transparent no matter what to see the vel channels
        if plotflux:
            img = ax.scatter3D(x2.value, z2, y2.value, c=flux2, cmap=gamma_color, alpha=alphav, s=mark_size, zorder=z3)
            fig.colorbar(img).set_label(hdr['bunit'], rotation=90, zorder=z3)
        else:
            ax.scatter3D(x2.value, z2, y2.value, c=z2, cmap=gamma_color, alpha=alphav, s=mark_size, zorder=z3)
        ###########################################
        ax.scatter3D(xstar, zstar, ystar,  marker='*', c=cstar, cmap=gamma_color, alpha=1, s=5, zorder=z3)
        ############################################
        ax.scatter3D(x1, z1, y1, c=cstar, alpha=1, s=1)  # para
        ax.set_ylim(zmax, zmin)
        ax.set_zlim(ymin.value, ymax.value)
        ax.set_xlim(xmin.value, xmax.value)
        ax.set_ylabel(zlabel)
        ax.set_xlabel('\u03B4 R.A. (\'\')')
        ax.set_zlabel('\u03B4 Dec. (\'\')')
        ax.set_title(legend)
        # plt.show()
        if vel < 10:
            plt.savefig(dir1 + "/"+ fname+'_vel_00'+str(vel)+'.png')
        elif vel >= 10 and vel < 100:
            plt.savefig(dir1 + "/"+fname+'_vel_0'+str(vel)+'.png')
        elif vel >= 100:
            plt.savefig(dir1 + "/"+fname+'vel_' + str(vel) + '.png')
        plt.close()

    print("---------------------------------------------------------------------------------------------------")
    print(f"{bcolors.OKGREEN}      [+] Creating the video{bcolors.ENDC}")
    movie_maker = "ffmpeg -y -loglevel panic -framerate 25  -i "+ dir1 +"/" +fname + "_vel_%03d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "+ dir1 + "/" + str(hdr['object']) + "_" + fname + "_VEL_" + str(rms) +".mp4"
 #   print(movie_maker)
    os.system(movie_maker)
    movie_maker = "ffmpeg -y -loglevel panic -i " + dir1 + "/" + str(hdr['object']) + "_" + fname + "_VEL_" + str(rms) + ".mp4 -filter:v \"setpts=4.0*PTS\" " + dir1 + "/" + str(hdr['object']) + "_" + fname + "_VEL_" + str(rms) +"_SLOW.mp4"
    os.system(movie_maker)
    # subprocess.run(["sh", "movie.sh"])
    # subprocess.run(["rm", "*.png"])

if keepfig:
    print("---------------------------------------------------------------------------------------------------")
    print(f"{bcolors.OKGREEN}      [+] Keeping figs  {bcolors.ENDC}" )
    print("---------------------------------------------------------------------------------------------------")
else:
    print("---------------------------------------------------------------------------------------------------")
    delete = "rm " + dir1 +"/" + fname + "*.png"
    print(f"{bcolors.OKGREEN}      [+] Deleting pngs:  {bcolors.ENDC}" + delete )
    os.system(delete)
    print("---------------------------------------------------------------------------------------------------")


print("---------------------------------------------------------------------------------------------------")
print(f"{bcolors.OKGREEN}[+] DONE. QUITING {bcolors.ENDC}")
print("---------------------------------------------------------------------------------------------------")


exit()
