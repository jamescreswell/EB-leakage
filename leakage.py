#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

import matplotlib as mpl

mpl.rcParams['pgf.rcfonts'] = False
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True # ticks to use math font, I think
#mpl.rcParams['font.serif'] = 'Times New Roman'
#plt.rcParams['axes.unicode_minus'] = True
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['legend.handlelength'] = 2
#mpl.rcParams['legend.handleheight'] = 0.8
mpl.rcParams['legend.handletextpad'] = 0.5
mpl.rcParams['legend.columnspacing'] = 1.0
mpl.rcParams['font.size'] = 0.0
mpl.rcParams['axes.titlepad'] = 1.0
mpl.rcParams['legend.fontsize'] = 6

textwidth = 7.0



belts = hp.read_map('fig2_belt.fits', field=(0,1,2,3,4,5,6,7,8))
mask = np.zeros(len(belts[0]))
mask[np.where(belts[0] != 0)[0]] = 1

masked_belts = []
for i in range(9):
    masked_belts.append(np.ma.masked_array(belts[i], -1*(1-mask), fill_value=np.nan))



cmap = plt.cm.jet
cmap.set_bad(color=(1, 1, 1, 0))



order = [2, 0, 1, 7, 6, 8, 4, 3, 5]
titles = ['Real leakage term', 'Real B map', 'Corrupted B map', 'Template (method 1)', 'Fixed B map (method 1)',
          'Residual (method 1)', 'Template (method 2)', 'Fixed B map (method 2)', 'Residual (method 2)']

ims = []
for i in range(9):
    ims.append(hp.cartview(masked_belts[order[i]], 1, sub=(3,3,i+1), lonra=[-10, 10], latra=[-2, 2], cmap=cmap, min=-0.6, max=0.6, cbar=False, return_projected_map=True, xsize=1500))




textwidth=6.9 * 0.48 * 2

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(textwidth, textwidth/3.05))

subplots = range(331, 340)
i = 0
for ax in axes.flat:
    im = ax.imshow(ims[i], vmin=-0.6, vmax=0.6, cmap=cmap, origin='lower')
    ax.set_title(titles[i], pad=-2.5)
    ax.set_axis_off()
    i = i + 1
    
image = ax.get_images()[-1]

cb = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', aspect=35*3.2/1.25, shrink=1.2, pad=0.02, ticks=[-0.6,  0.6], anchor=(0.6, -0.3))
cb.set_label(r"$\mu$K", labelpad=-7.9)

plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.18, wspace=0.1)

plt.savefig('disks_figure/leakage_belts_grid_600dpi_new.pdf', dpi=600)


disks = hp.read_map('fig2_disk.fits', field=(0,1,2,3,4,5,6,7,8))


mask = np.zeros(len(disks[0]))
mask[np.where(disks[0] != 0)[0]] = 1


masked_disks = []
for i in range(9):
    masked_disks.append(np.ma.masked_array(disks[i], -1*(1-mask), fill_value=np.nan))

order = [2, 0, 1, 7, 6, 8, 4, 3, 5]
titles = ['Real leakage term', 'Real B map', 'Corrupted B map', 'Template (1)', 'Fixed B map (1)',
          'Residual (1)', 'Template (2)', 'Fixed B map (2)', 'Residual (2)']

ims = []
for i in range(9):
    ims.append(hp.cartview(masked_disks[order[i]], 1, sub=(3,3,i+1), lonra=[-20, 20], latra=[-21.5, 21.5], cmap=cmap, min=-0.6, max=0.6, cbar=False, return_projected_map=True))



textwidth=6.9 * 0.48

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(textwidth, textwidth*1.15))

subplots = range(331, 340)
i = 0
for ax in axes.flat:
    im = ax.imshow(ims[i], vmin=-0.6, vmax=0.6, cmap=cmap, origin='lower')
    ax.set_title(titles[i])
    ax.set_axis_off()
    i = i + 1


cb = fig.colorbar(image, ax=axes.ravel().tolist(), orientation='horizontal', aspect=35, shrink=1.0, pad=0.05, ticks=[-0.6, 0.6], anchor=(0, -0.75))
cb.set_label(r"$\mu$K", labelpad=-7.9)


plt.subplots_adjust(left=0.05, right=0.95, top=0.99, bottom=0.1, hspace=0.001)

plt.savefig('disks_figure/leakage_disks_grid_600dpi_new.pdf', dpi=600)




BO = hp.read_map('in/new/BO.fits')
BR = hp.read_map('in/new/BR.fits')
BRw = hp.read_map('in/BRw.fits')
BT = hp.read_map('in/new/BT.fits')
win = hp.read_map('in/new/w.fits')



mask = np.zeros(len(BO))
mask[np.where(BO != 0)[0]] = 1



masked_disks = []
masked_disks.append(np.ma.masked_array(BO, -1*(1-mask), fill_value=np.nan))
masked_disks.append(np.ma.masked_array(BR, -1*(1-mask), fill_value=np.nan))
masked_disks.append(np.ma.masked_array(BRw, -1*(1-mask), fill_value=np.nan))
masked_disks.append(np.ma.masked_array(BT, -1*(1-mask), fill_value=np.nan))
masked_disks.append(np.ma.masked_array(win, -1*(1-mask), fill_value=np.nan))


order = [0, 3, 1, 4, 2]
titles = ['Corrupted', 'Template', 'Residual ' + r'$\left(\times 10\right)$', 'Apodization', 'Residual\n' + r'$\left(\times 10\right)$']



ims = []
for i in range(3):
    ims.append(hp.gnomview(masked_disks[order[i]], 1, sub=(3,3,i+1), reso=20, xsize=400, notext=True, cmap=cmap, min=-0.3, max=0.3, cbar=False, return_projected_map=True))



im3 = hp.gnomview(masked_disks[order[3]], 1, sub=(3,3,i+1), reso=10, xsize=800, notext=True, cmap='Greys', min=-0.3, max=0.3, cbar=False, return_projected_map=True)



textwidth=6.9 * 0.48

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(textwidth, textwidth/2.2))


i = 0
for ax in axes.flat:
    if i == 3:
        im3 = ax.imshow(ims[i], vmin=0, vmax=1.0, cmap=cmap, origin='lower')
    else:
        im = ax.imshow(ims[i], vmin=-0.3, vmax=0.3, cmap=cmap, origin='lower')
    if i == 2:
        t = ax.set_title(titles[i])
    else:
        ax.set_title(titles[i])
    ax.set_axis_off()
    i = i + 1

plt.setp(t, multialignment='center')

cb = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', aspect=35, shrink=0.99, pad=0.05, ticks=[-0.3, 0.3], anchor=(-0.96, 0.7))
cb.set_label(r"$\mu$K", labelpad=-7.9)


plt.subplots_adjust(left=0.05, right=0.95, bottom=0.3, hspace=0.001, wspace=0.01)

plt.savefig('disks_figure/zero_input_B_mode_new_600dpi.pdf', dpi=600)


im3 = plt.imshow(ims[3], vmin=0, vmax=1.0, cmap=cmap.set_under('None'), origin='lower')
plt.gca().set_axis_off()
plt.savefig('apo.png', bbox_inches='tight')




BO_cls = hp.anafast(BO*mask, lmax=1500)



l = np.arange(0, 1501)



BO_dls = BO_cls * l * (l+1) / (2 * np.pi)



BRw_cls = hp.anafast(BRw, lmax=1500)


BRw_dls = BRw_cls * l * (l+1) / (2 * np.pi)


plt.semilogy(l[1:], BO_dls[1:])
plt.semilogy(l[1:], BRw_dls[1:])
plt.ylim(1e-15, 1e2)



ps1 = np.fromfile('in/new/new/ps1.bin', dtype='float64')[:-1]
ps2 = np.fromfile('in/new/new/ps2.bin', dtype='float64')[:-1]
ps3 = np.fromfile('in/new/new/ps3.bin', dtype='float64')[:-1]
ps4 = np.fromfile('in/new/new/ps4.bin', dtype='float64')[:-1]
ps5 = np.fromfile('in/new/new/ps5.bin', dtype='float64')[:-1]


l = np.arange(1, len(ps1)+1)



ps1_binned = (ps1[0::4] + ps1[1::4] + ps1[2::4] + ps1[3::4]) / 4
ps2_binned = (ps2[0::4] + ps2[1::4] + ps2[2::4] + ps2[3::4]) / 4
ps3_binned = (ps3[0::4] + ps3[1::4] + ps3[2::4] + ps3[3::4]) / 4
ps4_binned = (ps4[0::4] + ps4[1::4] + ps4[2::4] + ps4[3::4]) / 4
ps5_binned = (ps5[0::4] + ps5[1::4] + ps5[2::4] + ps5[3::4]) / 4


l_binned = (l[0::4] + l[1::4] + l[2::4] + l[3::4]) / 4





import matplotlib.transforms as mtransforms
from matplotlib.patches import FancyBboxPatch

textwidth=6.9 * 0.48
fig = plt.figure(figsize=(textwidth, textwidth/2))


ax1 = plt.gca()

skip = 4

h = [
    plt.loglog(l_binned, ps1_binned, '#222222', label='$EE$', linewidth=0.75)[0],
]

h2 = [
    plt.loglog(l_binned, ps2_binned, 'orangered', label='Corrupted', linewidth=0.75)[0],
    plt.loglog(l_binned, ps3_binned, 'orangered', linestyle='--', label='(apodized)', linewidth=0.75)[0],
    plt.loglog(l_binned, ps4_binned, 'steelblue', label='Corrected', linewidth=0.75)[0],
    plt.loglog(l_binned, ps5_binned, 'steelblue',linestyle='--', label='(apodized)', linewidth=0.75)[0],
]


#h.insert(1, plt.plot([],[],color=(0,0,0,0), label=" ")[0])

plt.ylim(1e-14, 7e1)
plt.xlim(2, 1500)
plt.yticks([10**-10, 10**-5, 10**0])
plt.ylabel(r'$\widetilde{\mathcal{D}}_\ell$ $(\mu K^2)$')
plt.xlabel(r'$\ell$')
#leg0 = plt.legend(handles=[ps1plt])
leg0 = plt.legend(handles=h, bbox_to_anchor=(0.5965, 0.175), loc='lower right', frameon=False)
plt.gca().add_artist(leg0)
#g = plt.grid(linestyle='--', linewidth=0.5)


leg = plt.legend(handles=h2, bbox_to_anchor=(1.02, -0.025), loc='lower right', ncol=2, frameon=False)
plt.gca().add_artist(leg)
#plt.legend(loc='lower right')
#leg.get_frame().set_edgecolor('w')

#plt.legend(loc='lower left')

plt.tick_params(direction='in', which='both', right=True) 

# Create another legend for the second line.
plt.legend(handles=[], borderpad=3.3, labelspacing=1.5, columnspacing=2.5, bbox_to_anchor=(0.254, -0.01), loc='lower right')



newax = fig.add_axes([0.05, 0.12, 0.35, 0.35], anchor='NE', zorder=+1)
im33 = newax.imshow(im3, vmin=0, vmax=1, cmap='jet')
cb = fig.colorbar(im33, ax=newax, orientation='horizontal', aspect=9, shrink=0.33, pad=0.05, ticks=[], anchor=(0.5, 1.35))
newax.axis('off')
#newax.set_title('Apodization')
newax.set_xticks([])
newax.set_yticks([])



plt.savefig('disks_figure/zero_B_ps.pdf', bbox_inches='tight', dpi=600)



print mpl.rcParams['legend.fancybox']
print mpl.rcParams['legend.loc']
print mpl.rcParams['legend.numpoints']
print mpl.rcParams['legend.fontsize'] 
print mpl.rcParams['legend.framealpha']
print mpl.rcParams['legend.scatterpoints'] 
print mpl.rcParams['legend.edgecolor']
print mpl.rcParams['xtick.top']
print mpl.rcParams['ytick.right']



def draw_bbox(ax, bb):
    # boxstyle=square with pad=0, i.e. bbox itself.
    p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                            abs(bb.width), abs(bb.height),
                            boxstyle="square,pad=0.",
                            ec="k", fc="none", zorder=10.,
                            )
    ax.add_patch(p_bbox)


def test1(ax):

    # a fancy box with round corners. pad=0.1
    p_fancy = FancyBboxPatch((bb.xmin, bb.ymin),
                             abs(bb.width), abs(bb.height),
                             boxstyle="round,pad=0.1",
                             fc=(1., .8, 1.),
                             ec=(1., 0.5, 1.))

    ax.add_patch(p_fancy)

    ax.text(0.1, 0.8,
            r' boxstyle="round,pad=0.1"',
            size=10, transform=ax.transAxes)

    draw_bbox(ax, bb)








x = hp.read_map('fig2_disk.fits', field=5)




plt.figure(figsize=(20, 20))
hp.cartview(masked_belts[3], 1, lonra=[-10, 10], latra=[-2, 2], cmap=cmap, min=-0.6, max=0.6)



def get_edge(mask):
    edge = []
    for i in range(len(mask)):
        if mask[i] == 1:
            neighbours = hp.get_all_neighbours(hp.npix2nside(len(mask)), i)
            edge_pixel = False
            for neighbour in neighbours:
                if neighbour != -1:
                    if mask[neighbour] == 0:
                        edge_pixel = True
            if edge_pixel:
                edge.append(i)
    return np.array(edge)



edge = get_edge(mask)



x = np.ones(len(mask))
x[edge] = 0 
hp.cartview(x, lonra=[-30, 30], latra=[-25, 10])



hp.mollview(mask)


def neighbours_matrix(nside):
    matrix = []
    for i in range(hp.nside2npix(nside)):
        matrix.append(hp.get_all_neighbours(nside, i))
    return np.array(matrix)

def inpaint(m, mask, N, neighbours, valid_sky, edge, interior):
    #print 'Calculating neighbours...\r'
    #neighbours = neighbours_matrix(nside)
    
    #print 'Calculating mask edge and interior...\r'
    
    #valid_sky = np.where(mask==1)[0]
    #edge = get_edge(mask)
    #interior = np.setdiff1d(valid_sky, edge)
    
    #print 'Copying map...\r'
    
    template = m.copy()
    template = template*mask
    
    #print 'Inpainting...'
    
    for i in range(N):
        template[interior] = np.mean(template[neighbours[interior]], axis=1)
        #print '%d/%d\r' % (i+1, N),
    
    return template


# In[13]:


template = inpaint(maskedB, mask, 1000)


# In[5]:


def linearly_remove(m, mask, template):
    valid_sky = np.where(mask==1)[0]
    edge = get_edge(mask)
    interior = np.setdiff1d(valid_sky, edge)
    
    alphas = np.linspace(0, 3, num=1000)
    
    res = np.array([np.sum(np.abs(m[interior] - alpha * template[interior])**2) for alpha in alphas])
    
    loc = alphas[np.argmin(res)]
    
    return m - loc*template, loc


# In[15]:


correctedB, x = linearly_remove(maskedB, mask, template)


# In[16]:


hp.cartview(correctedB*mask - B*mask, cmap=cmap, lonra=[-30, 30], latra=[-25, 10], min=-2.5e-6, max=2.5e-6)
plt.title('Corrected difference')
hp.cartview(maskedB*mask - B*mask, cmap=cmap, lonra=[-30, 30], latra=[-25, 10], min=-2.5e-6, max=2.5e-6)
plt.title('Original difference')


# In[6]:


def method1(m, mask, N, neighbours, valid_sky, edge, interior):
    # m is full sky [T, Q, U]
    # Returns: true B, corrupted B, corrected B, alpha
    T, E, B = tqu2teb(m[0], m[1], m[2])
    
    maskedT, maskedE, maskedB = tqu2teb(m[0]*mask, m[1]*mask, m[2]*mask)
    
    template = inpaint(maskedB, mask, N, neighbours, valid_sky, edge, interior)
    
    correctedB, alpha = linearly_remove(maskedB, mask, template)
    
    return B*mask, maskedB*mask, correctedB*mask, alpha





def tqu2qeueqbub(T, Q, U, getEfamily=True, getBfamily=True):
    nside = hp.npix2nside(len(T))
    
    alms = hp.map2alm([T, Q, U], lmax=int(2.5*nside-1), pol=True)
    alms0 = np.zeros(len(alms[0]), dtype='complex')
    
    if getEfamily:
        Efamily = hp.alm2map([alms[0], alms[1], alms0], nside=nside, lmax=int(2.5*nside-1), pol=True, verbose=False)
        QE = Efamily[1]
        UE = Efamily[2]
    if getBfamily:
        Bfamily = hp.alm2map([alms[0], alms0, alms[2]], nside=nside, lmax=int(2.5*nside-1), pol=True, verbose=False)
        QB = Bfamily[1]
        UB = Bfamily[2]
        
    if getEfamily and getBfamily:
        return QE, UE, QB, UB
    elif getEfamily:
        return QE, UE
    elif getBfamily:
        return QB, UB


def method2(m, mask):
    # m is full sky [T, Q, U]
    # Returns: true B, corrupted B, corrected B, alpha
    T, E, B = tqu2teb(m[0], m[1], m[2])
    
    maskedT, maskedE, maskedB = tqu2teb(m[0]*mask, m[1]*mask, m[2]*mask)
    QEp, UEp = tqu2qeueqbub(m[0]*mask, m[1]*mask, m[2]*mask, getBfamily=False)
    
    QEp = QEp * mask
    UEp = UEp * mask
    
    _, _, template = tqu2teb(m[0]*mask, QEp, UEp)
    
    correctedB, alpha = linearly_remove(maskedB, mask, template)
    
    return B*mask, maskedB*mask, correctedB*mask, alpha



res = method2(ffp9, mask)




hp.cartview(res[0], lonra=[-30, 30], latra=[-25, 10], cmap=cmap, min=-2e-6, max=2.5e-6)
hp.cartview(res[1], lonra=[-30, 30], latra=[-25, 10], cmap=cmap, min=-2e-6, max=2.5e-6)
hp.cartview(res[2], lonra=[-30, 30], latra=[-25, 10], cmap=cmap, min=-2e-6, max=2.5e-6)




dx12 = ffp9


dx12 = [hp.ud_grade(dx12[i], nside_out=nside) for i in range(3)]




T, E, B = tqu2teb(dx12[0], dx12[1], dx12[2])



Ealm = hp.map2alm(E, lmax=int(2.5*nside-1))
Talm = hp.map2alm(T, lmax=int(2.5*nside-1))
Balm = np.zeros(len(Ealm), dtype='complex')

dx12 = hp.alm2map([Talm, Ealm, Balm], pol=True, nside=nside, lmax=int(2.5*nside-1))
T, E, B = tqu2teb(dx12[0], dx12[1], dx12[2])



nside=128
mask = np.zeros(hp.nside2npix(nside))

disc1 = hp.query_disc(nside, hp.ang2vec(np.pi/2, 0.0), 47.41*np.pi/180.0)

mask[disc1] = 1.0

hp.mollview(mask)




res = method2(dx12, mask)




res0 = np.ma.masked_array(res[0], -1*(1-mask), fill_value=np.nan) 
res1 = np.ma.masked_array(res[1], -1*(1-mask), fill_value=np.nan) 
res2 = np.ma.masked_array(res[2], -1*(1-mask), fill_value=np.nan) 
cmap.set_bad('w')

hp.cartview(res0, lonra=[-50, 50], latra=[-50, 50], cmap=cmap, min=-2e-6, max=2.5e-6)
plt.title('Real B mode')
hp.cartview(res1, lonra=[-50, 50], latra=[-50, 50], cmap=cmap, min=-2e-6, max=2.5e-6)
plt.title('Corrupted B mode')
hp.cartview(res2, lonra=[-50, 50], latra=[-50, 50], cmap=cmap, min=-2e-6, max=2.5e-6)
plt.title('Corrected B mode')



EE = hp.anafast(E*mask*win)
BB_corrupted = hp.anafast(res[1]*win)
BB_corrected = hp.anafast(res[2]*win)
ell = np.arange(2, len(EE)+2)



plt.semilogy(ell, EE*ell*(ell+1), label='EE')
plt.semilogy(ell, BB_corrupted*ell*(ell+1), label='corrupted BB')
plt.semilogy(ell, BB_corrected*ell*(ell+1), label='corrected BB')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$D_\ell$')
plt.legend()
plt.xlim(1,2*nside-1)
plt.ylim(1e-25, 1e-10)




alphas4 = []
for i in range(25):
    ngc = hp.read_map('in/FFP9/ngc/n0128/ffp9_cmb_ngc_353_full_map_mc_' + str(i).zfill(4) + '_n0128.fits', field=(0, 1, 2), verbose=False)
    scl = hp.read_map('in/FFP9/scl/n0128/ffp9_cmb_scl_353_full_map_mc_' + str(i).zfill(4) + '_n0128.fits', field=(0, 1, 2), verbose=False)
    ten = hp.read_map('in/FFP9/ten/n0128/ffp9_cmb_ten_353_full_map_mc_' + str(i).zfill(4) + '_n0128.fits', field=(0, 1, 2), verbose=False)

    ffp9 = scl + np.sqrt(0.05 / 0.1) * ten + 5 * ngc

    res = method2(ffp9, mask)
    alphas4.append(res[3])




plt.plot(alphas)
plt.plot(alphas2)
plt.plot(alphas3)
plt.plot(alphas4)


# ## Window functions


nside=256




def make_window(window, mask):
    nside = hp.npix2nside(len(mask))
    
    valid_sky = np.where(mask==1)[0]
    edge = get_edge(mask)
    interior = np.setdiff1d(valid_sky, edge)
    
    distance = np.zeros(len(mask))
    i = 0
    for pixel in interior:
#         min_distance = 1000
#         vec = hp.pix2vec(nside, pixel)
#         for edge_pixel in edge:
#             dist = np.arccos(np.dot(vec, hp.pix2vec(nside, edge_pixel)))
#             if dist < min_distance:
#                 min_distance = dist
#         distance[pixel] = min_distance
        vec = hp.pix2vec(nside, pixel)
        distance[pixel] = 47.41*np.pi/180.0 - np.arccos(np.dot(vec, hp.ang2vec(np.pi/2, 0.0)))
        #i = i + 1
        #print "%d/%d\r" % (i, len(interior)),
    
    max_distance = np.max(distance)
    
    map_window = np.zeros(len(mask))
    for pixel in interior:
        arg = int((distance[pixel] / (2.0 * max_distance)) * len(window))
        map_window[pixel] = window[arg]
    return map_window



res = make_window(np.hanning(1000), mask)



hp.mollview(res)



hp.mollview(res)


neighbours=neighbours_matrix(nside)
valid_sky = np.where(mask==1)[0]
edge = get_edge(mask)
interior=np.setdiff1d(valid_sky, edge)


# In[13]:


def R(map_window, m, mask, l1, l2):
    realB, corruptedB, correctedB, alpha = method1(m, mask, 500, neighbours, valid_sky, edge, interior)
    
    BB_real = hp.anafast(realB * map_window)
    #BB_corrupted = hp.anafast(corruptedB*map_window)
    BB_corrected = hp.anafast(correctedB*map_window)
    #ell = np.arange(2, len(EE)+2)
    
    #oldR = np.sqrt((1.0/(l2-l1+1))*np.sum((BB_real - BB_corrected)[l1:l2]**2)) / ((1.0/(l2-l1+1))*np.sum(BB_real[l1:l2]))
    R = np.sum(((BB_real - BB_corrected)[l1:l2] / (BB_real[l1:l2]))**2)
    
    return R

def f(map_window):
    return 1.0*len(np.where(map_window>0)[0])/np.sqrt(np.sum(map_window**2))




def make_Planck_mask(mask):
    m = np.copy(mask)
    for i in range(3):
        m = hp.smoothing(m, fwhm=5.0*np.pi/180.0)
        m[np.where(m < 0.5)[0]] = 0
        m[np.where(m != 0)[0]] = (m[np.where(m != 0)[0]] - 0.5) * (1.0 / (1.0 - 0.5))
    return m



from scipy.signal import get_window

def run_simulation(window_map, mask):
    Rsum = 0
    for i in range(50):
        ngc = hp.read_map('in/FFP9/ngc/n0128/ffp9_cmb_ngc_353_full_map_mc_' + str(i).zfill(4) + '_n0128.fits', field=(0, 1, 2), verbose=False)
        scl = hp.read_map('in/FFP9/scl/n0128/ffp9_cmb_scl_353_full_map_mc_' + str(i).zfill(4) + '_n0128.fits', field=(0, 1, 2), verbose=False)
        ten = hp.read_map('in/FFP9/ten/n0128/ffp9_cmb_ten_353_full_map_mc_' + str(i).zfill(4) + '_n0128.fits', field=(0, 1, 2), verbose=False)

#         ngc = np.array([hp.ud_grade(ngc[i], nside) for i in range(3)])
#         scl = np.array([hp.ud_grade(scl[i], nside) for i in range(3)])
#         ten = np.array([hp.ud_grade(ten[i], nside) for i in range(3)])
        
        ffp9 = scl + np.sqrt(0.05 / 0.1) * ten + 5 * ngc
    
        Rsum = Rsum + R(window_map, ffp9, mask, 60, 120)
        
        print "%d\r" % i,
    
    return np.sqrt(1.0/(100.0 * (120-60+1.0)) * Rsum)



Rs = run_simulation(make_Planck_mask(mask), mask)



print Rs
print newf(make_Planck_mask(mask))




tukeymap01 = make_window(get_window(('tukey', 0.1), 1000), mask)

Rs_tukeymap01 = run_simulation(tukeymap01, mask)



tukeymap02 = make_window(get_window(('tukey', 0.2), 1000), mask)

Rs_tukey02 = run_simulation(tukeymap02, mask)



tukeymap03 = make_window(get_window(('tukey', 0.3), 1000), mask)

Rs_tukey03 = run_simulation(tukeymap03, mask)



tukeymap04 = make_window(get_window(('tukey', 0.4), 1000), mask)

Rs_tukey04 = run_simulation(tukeymap04, mask)



tukeymap05 = make_window(get_window(('tukey', 0.5), 1000), mask)

Rs_tukey05 = run_simulation(tukeymap05, mask)



tukeymap06 = make_window(get_window(('tukey', 0.6), 1000), mask)

Rs_tukey06 = run_simulation(tukeymap06, mask)



tukeymap07 = make_window(get_window(('tukey', 0.7), 1000), mask)

Rs_tukey07 = run_simulation(tukeymap07, mask)



tukeymap08 = make_window(get_window(('tukey', 0.8), 1000), mask)

Rs_tukey08 = run_simulation(tukeymap08, mask)


tukeymap09 = make_window(get_window(('tukey', 0.9), 1000), mask)

Rs_tukey09 = run_simulation(tukeymap09, mask)



tukeymap10 = make_window(get_window(('tukey', 1.0), 1000), mask)

Rs_tukey10 = run_simulation(tukeymap10, mask)



hammingmap = make_window(get_window('hamming', 1000), mask)

Rs_hamming = run_simulation(hammingmap, mask)


bartlettmap = make_window(get_window('bartlett', 1000), mask)

Rs_bartlett = run_simulation(bartlettmap, mask)

nuttallmap = make_window(get_window('nuttall', 1000), mask)

Rs_nuttall = run_simulation(nuttallmap, mask)



blackmanmap = make_window(get_window('blackman', 1000), mask)

Rs_blackman = run_simulation(blackmanmap, mask)



def f(map_window):
    return 1.0*len(np.where(map_window>0)[0])/np.sqrt(np.sum(map_window**2))

def newf(windowmap):
    return (1.0/len(np.where(windowmap>0)[0])) * np.sum(windowmap**2)


import matplotlib as mpl

mpl.rcParams['pgf.rcfonts'] = False
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True # ticks to use math font, I think

#plt.rcParams['axes.unicode_minus'] = True

mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handleheight'] = 1
mpl.rcParams['legend.handletextpad'] = 0.5
mpl.rcParams['legend.columnspacing'] = 1.0
mpl.rcParams['axes.titlepad'] = 6.0

mpl.rcParams['xtick.top'] = False
mpl.rcParams['ytick.right'] = False




textwidth= 6.9 * 0.48 * 2

plt.figure(figsize=(textwidth/2, textwidth/3*2))

plt.subplot(2, 1, 1)

#plt.axvline(1e-4, linestyle='--', color='gray', linewidth=1.25)

ps = 15
g = plt.grid(linestyle='--', linewidth=0.5)

ax = plt.gca()
ax.set_axisbelow(True)


plt.scatter(f_tukeymap01, Rs_tukey01, label='tu01', color='black', s=ps)
plt.scatter(f_tukeymap02, Rs_tukey02, label='tu02', color='black', s=ps)
plt.scatter(f_tukeymap03, Rs_tukey03, label='tu03', color='black', s=ps)
plt.scatter(f_tukeymap04, Rs_tukey04, label='tu04', color='black', s=ps)
plt.scatter(f_tukeymap05, Rs_tukey05, label='tu05', color='black', s=ps)
plt.scatter(f_tukeymap06, Rs_tukey06, label='tu06', color='black', s=ps)
plt.scatter(f_tukeymap07, Rs_tukey07, label='tu07', color='black', s=ps)
plt.scatter(f_tukeymap08, Rs_tukey08, label='tu08', color='black', s=ps)
plt.scatter(f_tukeymap09, Rs_tukey09, label='tu09', color='black', s=ps)
plt.scatter(f_tukeymap10, Rs_tukey10, label='tu10', color='black', s=ps)

plt.scatter(f_hamming, Rs_hamming, label='ha', color='red', s=ps)
plt.scatter(f_bartlett, Rs_bartlett, label='ba', color='steelblue', s=ps)
plt.scatter(f_nuttall, Rs_nuttall, label='nu', color='seagreen', s=ps)
plt.scatter(f_blackman, Rs_blackman, label='bl', color='orange', s=ps)

fs = 7

diffx = -17#-20
diffy = 14#2
rot = -45

plt.annotate('tu0.1', xy=(f_tukeymap01, Rs_tukey01), xytext=(diffx, diffy),textcoords='offset points', fontsize=fs, rotation=rot)
plt.annotate('tu0.2', xy=(f_tukeymap02, Rs_tukey02), xytext=(diffx, diffy),textcoords='offset points', fontsize=fs, rotation=rot)
plt.annotate('tu0.3', xy=(f_tukeymap03, Rs_tukey03), xytext=(diffx, diffy),textcoords='offset points', fontsize=fs, rotation=rot)
plt.annotate('tu0.4', xy=(f_tukeymap04, Rs_tukey04), xytext=(diffx, diffy),textcoords='offset points', fontsize=fs, rotation=rot)
plt.annotate('tu0.5', xy=(f_tukeymap05, Rs_tukey05), xytext=(diffx, diffy),textcoords='offset points', fontsize=fs, rotation=rot)
plt.annotate('tu0.6', xy=(f_tukeymap06, Rs_tukey06), xytext=(diffx, diffy),textcoords='offset points', fontsize=fs, rotation=rot)
plt.annotate('tu0.7', xy=(f_tukeymap07, Rs_tukey07), xytext=(diffx, diffy),textcoords='offset points', fontsize=fs, rotation=rot)
plt.annotate('tu0.8', xy=(f_tukeymap08, Rs_tukey08), xytext=(diffx, diffy),textcoords='offset points', fontsize=fs, rotation=rot)
plt.annotate('tu0.9', xy=(f_tukeymap09, Rs_tukey09), xytext=(diffx, diffy),textcoords='offset points', fontsize=fs, rotation=rot)
plt.annotate('tu1.0', xy=(f_tukeymap10, Rs_tukey10), xytext=(diffx, diffy),textcoords='offset points', fontsize=fs, rotation=rot)



plt.annotate('ha', xy=(f_hamming, Rs_hamming), xytext=(-10, 2),textcoords='offset points', fontsize=fs)
plt.annotate('ba', xy=(f_bartlett, Rs_bartlett), xytext=(-10, 2),textcoords='offset points', fontsize=fs)
plt.annotate('nu', xy=(f_nuttall, Rs_nuttall), xytext=(-12, 0),textcoords='offset points', fontsize=fs)
plt.annotate('bl', xy=(f_blackman, Rs_blackman), xytext=(-8, 2),textcoords='offset points', fontsize=fs)

plt.tick_params(direction='in', which='both', right=True, top=True) 


#plt.legend()
plt.ylabel(r'$R$')
plt.xlabel(r'$f_W$')
plt.title('Method 1')

plt.gca().set_yscale('log')
plt.ylim(2e-5, 3.2e-2)
plt.xlim(0, 1)



plt.subplot(2, 1, 2)

plt.axhline(f_hamming/Rs_hamming, color='red', label='ha', linewidth=1.25)
plt.axhline(f_bartlett/Rs_bartlett, color='steelblue', label='ba', linewidth=1.25)
plt.axhline(f_nuttall/Rs_nuttall, color='seagreen', label='nu', linewidth=1.25)
plt.axhline(f_blackman/Rs_blackman, color='orange', label='bl', linewidth=1.25)

plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [
    1/(Rs_tukey01/f_tukeymap01),
    1/(Rs_tukey02/f_tukeymap02),
    1/(Rs_tukey03/f_tukeymap03),
    1/(Rs_tukey04/f_tukeymap04),
    1/(Rs_tukey05/f_tukeymap05),
    1/(Rs_tukey06/f_tukeymap06),
    1/(Rs_tukey07/f_tukeymap07),
    1/(Rs_tukey08/f_tukeymap08),
    1/(Rs_tukey09/f_tukeymap09),
    1/(Rs_tukey10/f_tukeymap10)], 'o', color='k', label='tu', markersize=np.sqrt(ps), linewidth=1.25)

g = plt.grid(linestyle='--', linewidth=0.5)


plt.ylim(0, 4200)
plt.tick_params(direction='in', which='both', right=True, top=True) 

#plt.legend(loc='lower right', ncol=3, bbox_to_anchor=[1.0,0.6])
plt.legend(loc='upper left', ncol=3)
#plt.gca().set_yscale('log')
plt.xlabel('Tukey taper fraction')
plt.ylabel(r'$f_W/R$')

plt.subplots_adjust(hspace=0.4)


plt.savefig('method1.pdf', bbox_inches='tight')


# In[38]:


Rs_tukey01, f_tukeymap01 = 0.00814990792528, 0.894634287668
Rs_tukey02, f_tukeymap02 = 0.00205139586594, 0.785458312294
Rs_tukey03, f_tukeymap03 = 0.000932171856032, 0.68312410723
Rs_tukey04, f_tukeymap04 = 0.000538260827417, 0.587939256722
Rs_tukey05, f_tukeymap05 = 0.000360018054963, 0.500214568072
Rs_tukey06, f_tukeymap06 = 0.000261534910353, 0.42023185699
Rs_tukey07, f_tukeymap07 = 0.00020338094709, 0.34823573729
Rs_tukey08, f_tukeymap08 = 0.000166892564813, 0.284444392111
Rs_tukey09, f_tukeymap09 = 0.000144100979142, 0.229047867493
Rs_tukey10, f_tukeymap10 = 0.000128956883981, 0.182203283896


Rs_hamming = 0.00464137905575
f_hamming = 0.206352127684
Rs_bartlett = 0.00128667892637
f_bartlett = 0.175045532027
Rs_nuttall = 3.86059025534e-05
f_nuttall = 0.0919547689048
Rs_blackman = 5.695790964e-05
f_blackman = 0.123808892829


# In[180]:


print 'method1'

print Rs_tukeymap01, newf(tukeymap01)
print Rs_tukey02, newf(tukeymap02)
print Rs_tukey03, newf(tukeymap03)
print Rs_tukey04, newf(tukeymap04)
print Rs_tukey05, newf(tukeymap05)
print Rs_tukey06, newf(tukeymap06)
print Rs_tukey07, newf(tukeymap07)
print Rs_tukey08, newf(tukeymap08)
print Rs_tukey09, newf(tukeymap09)
print Rs_tukey10, newf(tukeymap10)

print Rs_hamming, newf(hammingmap)
print Rs_bartlett, newf(bartlettmap)
print Rs_nuttall, newf(nuttallmap)
print Rs_blackman, newf(blackmanmap)


# In[113]:


plt.figure(figsize=(6.5,4))
plt.plot([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [
    1/(np.mean(Rs_tukey02)/newf(tukeymap02)),
    1/(np.mean(Rs_tukey03)/newf(tukeymap03)),
    1/(np.mean(Rs_tukey04)/newf(tukeymap04)),
    1/(np.mean(Rs_tukey05)/newf(tukeymap05)),
    1/(np.mean(Rs_tukey06)/newf(tukeymap06)),
    1/(np.mean(Rs_tukey07)/newf(tukeymap07)),
    1/(np.mean(Rs_tukey08)/newf(tukeymap08)),
    1/(np.mean(Rs_tukey09)/newf(tukeymap09))])

plt.scatter([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [
    1/(np.mean(Rs_tukey02)/newf(tukeymap02)),
    1/(np.mean(Rs_tukey03)/newf(tukeymap03)),
    1/(np.mean(Rs_tukey04)/newf(tukeymap04)),
    1/(np.mean(Rs_tukey05)/newf(tukeymap05)),
    1/(np.mean(Rs_tukey06)/newf(tukeymap06)),
    1/(np.mean(Rs_tukey07)/newf(tukeymap07)),
    1/(np.mean(Rs_tukey08)/newf(tukeymap08)),
    1/(np.mean(Rs_tukey09)/newf(tukeymap09))])

plt.xlabel('Taper fraction')
plt.ylabel(r'$f_W/R$')
plt.title('Method 1')





plt.figure(figsize=(6.5,4))

plt.scatter(np.mean(Rs_tukey02), newf(tukeymap02), label='tu02')
plt.scatter(np.mean(Rs_tukey03), newf(tukeymap03), label='tu03')
plt.scatter(np.mean(Rs_tukey04), newf(tukeymap04), label='tu04')
plt.scatter(np.mean(Rs_tukey05), newf(tukeymap05), label='tu05')
plt.scatter(np.mean(Rs_tukey06), newf(tukeymap06), label='tu06')
plt.scatter(np.mean(Rs_tukey07), newf(tukeymap07), label='tu07')
plt.scatter(np.mean(Rs_tukey08), newf(tukeymap08), label='tu08')
plt.scatter(np.mean(Rs_tukey09), newf(tukeymap09), label='tu09')
plt.legend()
plt.xlabel(r'$R$')
plt.ylabel(r'$f_W$')
plt.title('Method 2')

plt.gca().set_xscale('log')
plt.xlim(1e-4, 2e-3)

plt.savefig('method2-1.pdf', bbox_inches='tight')


# In[88]:


plt.figure(figsize=(6.5,4))
plt.plot([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [
    1/(np.mean(Rs_tukey02)/newf(tukeymap02)),
    1/(np.mean(Rs_tukey03)/newf(tukeymap03)),
    1/(np.mean(Rs_tukey04)/newf(tukeymap04)),
    1/(np.mean(Rs_tukey05)/newf(tukeymap05)),
    1/(np.mean(Rs_tukey06)/newf(tukeymap06)),
    1/(np.mean(Rs_tukey07)/newf(tukeymap07)),
    1/(np.mean(Rs_tukey08)/newf(tukeymap08)),
    1/(np.mean(Rs_tukey09)/newf(tukeymap09))])

plt.scatter([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [
    1/(np.mean(Rs_tukey02)/newf(tukeymap02)),
    1/(np.mean(Rs_tukey03)/newf(tukeymap03)),
    1/(np.mean(Rs_tukey04)/newf(tukeymap04)),
    1/(np.mean(Rs_tukey05)/newf(tukeymap05)),
    1/(np.mean(Rs_tukey06)/newf(tukeymap06)),
    1/(np.mean(Rs_tukey07)/newf(tukeymap07)),
    1/(np.mean(Rs_tukey08)/newf(tukeymap08)),
    1/(np.mean(Rs_tukey09)/newf(tukeymap09))])

plt.xlabel('Taper fraction')
plt.ylabel(r'$f_W/R$')
plt.title('Method 2')

plt.savefig('method2-2.pdf', bbox_inches='tight')

