import matplotlib.font_manager as fm
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import warnings,cv2,ffmpeg,glob,re,pprint,os,math
from matplotlib import rcParams
from matplotlib.ticker import StrMethodFormatter, FuncFormatter
from matplotlib.transforms import Bbox
from PIL import Image

class cfg_my_settings:
    def __init__(self,base_font_size=14,dpi=100):
        self.kb=1.3806488e-23
        self.mp=1.67262178e-27
        self.me=9.10938291e-31
        self.e=1.602176565e-19
        self.eps0=8.854187817e-12
        self.mu0=4e-7*math.pi
        self.ke=8.9875517873681764e9
        self.c=299792458
        self.mole=6.02214129e23
        self.r_earth=6371.2e3
        self.h_ev=4.135667696e-15
        self.h_p=6.62607015e-34
        
        self.base_font_size = base_font_size
        warnings.filterwarnings("ignore")
        
        rcParams['text.usetex'] = True
        rcParams['text.latex.preamble'] = r'\usepackage{bm} \boldmath \renewcommand{\rmdefault}{pbk} \bfseries'
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['American Typewriter']
        rcParams['font.weight'] = 'bold'
        rcParams['font.size'] = self.base_font_size
        
        rcParams['figure.dpi'] = dpi # can be commented, change dpi can alter relative fontsize and axis position
        rcParams['image.cmap'] = 'jet'
        rcParams['figure.figsize'] = [25.6,14.4] # can be commented
        rcParams['figure.autolayout'] = False
        rcParams['figure.constrained_layout.use'] = False
        # rcParams['figure.constrained_layout.wspace'] = 0.2
        # rcParams['figure.constrained_layout.hspace'] = 0.2
        rcParams['lines.linewidth'] = 2

        custom_colors = [
        'blue', 'red', 'green', 'black', 'orange', 'purple', 'brown', 
        'cyan', 'magenta', 'yellow', 'gray', 'olive', 'pink', 'teal', 
        'navy', 'maroon', 'lime', 'gold', 'indigo', 'turquoise' ]
        
        rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)
        rcParams['axes.linewidth'] = 1.5
        rcParams['axes.spines.top'] = True
        rcParams['axes.spines.bottom'] = True
        rcParams['axes.spines.left'] = True
        rcParams['axes.spines.right'] = True
        rcParams['axes.titlesize'] = self.base_font_size + 4
        rcParams['axes.labelsize'] = self.base_font_size + 2
        rcParams['axes.formatter.useoffset'] = True
        rcParams['axes.formatter.limits'] = (-3, 3) # can be commented
        rcParams['axes.titlepad'] = 2
        rcParams['axes.titleweight'] = 'bold'
        rcParams['axes.labelpad'] = 2
        
        rcParams['grid.linestyle'] = '--'
        rcParams['grid.linewidth'] = 0.5
        rcParams['grid.alpha'] = 0.7

        rcParams['xtick.direction'] = 'inout'
        rcParams['ytick.direction'] = 'inout'
        rcParams['xtick.major.size'] = 6
        rcParams['ytick.major.size'] = 6
        rcParams['xtick.major.width'] = 1.5
        rcParams['ytick.major.width'] = 1.5
        rcParams['xtick.labelsize'] = self.base_font_size - 1
        rcParams['ytick.labelsize'] = self.base_font_size - 1
        rcParams['xtick.major.pad'] = 1
        rcParams['ytick.major.pad'] = 1
        
        rcParams['legend.borderaxespad'] = 0.5
        rcParams['legend.frameon'] = False
        rcParams['legend.fontsize'] = self.base_font_size - 2
        rcParams['legend.labelspacing'] = 0.01
        rcParams['legend.borderpad'] = 0
        # rcParams['legend.handlelength'] = 1.0
        rcParams['legend.handletextpad'] = 0.1
        
        rcParams['savefig.dpi'] = 600

    def my_colorbar(self, cf, ax,  cb_width=0.03, cb_div=1.015,shrink=0.9, ticklabelsize = None): # call this only after every axis adjustment
        if ticklabelsize is None:
                ticklabelsize = self.base_font_size - 4
        pos1 = ax.get_position()
        cb_h = pos1.height*shrink
        cb_w = pos1.width*cb_width
        formatter = matplotlib.ticker.ScalarFormatter()
        formatter.set_powerlimits((0, 0))
        #cb = plt.colorbar(cf, ax=ax, format=FuncFormatter(StrMethodFormatter('{x:.2e}')), shrink=shrink , location='right' , aspect = cb_h /cb_w )
        cb = plt.colorbar(cf, ax=ax, format=formatter, shrink=shrink , location='right' , aspect = cb_h /cb_w )
        cb.outline.set_linewidth(1.5)
        cb.ax.tick_params(width=1.5, direction='inout', length=6, labelsize=ticklabelsize, pad=1)
        for l in cb.ax.get_yticklabels():
            l.set_fontweight('bold')
            l.set_fontsize(ticklabelsize)
        ax.set_position(pos1)
        #pos_cb = Bbox.from_bounds( pos1.x1*cb_div , pos1.y0+pos1.height*(1-shrink)*0.5 ,  cb_w , cb_h )
        pos_cb = Bbox.from_bounds( pos1.x1+pos1.width*0.3 , pos1.y0+pos1.height*(1-shrink)*0.5 ,  cb_w , cb_h )
        cb.ax.set_position(pos_cb)
        return cb
    
    def my_savefig(self, filename, dpi=600, bbox_inches='tight', **kwargs):
        plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, pil_kwargs={"quality": 95}, **kwargs)
    
    def generate_gif(self,folderpath, savename, ext='jpg'):
        pprint.pp("Generating GIF...")
        image_paths = sorted(glob.glob( re.sub(r'([\[\]])', r'[\1]', folderpath) + "*." + ext ))
        images = []
        output_anime = folderpath + savename + ".gif"
        for image_path1 in image_paths:
            images.append(Image.open(image_path1))
        with Image.open(image_paths[0]) as first_image:
            first_image.save(output_anime, save_all=True, append_images=images[1:], duration=100, loop=0, optimize=True)
        pprint.pp(output_anime)
    
    def generate_video_cv2(self,folderpath, savename, ext='jpg'):
        pprint.pp("Generating video (cv2)...")
        image_paths = sorted(glob.glob( re.sub(r'([\[\]])', r'[\1]', folderpath) + "*." + ext ))
        frame = cv2.imread(image_paths[0])
        ar = frame.shape[1] / frame.shape[0]
        output_anime = folderpath + savename + ".avi"
        video = cv2.VideoWriter(output_anime, cv2.VideoWriter_fourcc(*'VP90'), 10, (3840, int(3840 / ar)),isColor=True)
        for image_path1 in image_paths:
            frame = cv2.imread(image_path1)
            resized_frame = cv2.resize(frame, (3840, int(3840 / ar)))
            video.write(resized_frame)
        video.release()
        pprint.pp(output_anime)
    
    def generate_video_ffmpeg(self,folderpath, savename, ext='jpg'):
        pprint.pp("Generating video (ffmpeg)...")
        image_paths = sorted(glob.glob( re.sub(r'([\[\]])', r'[\1]', folderpath) + "*." + ext ))
        output_anime = folderpath + savename + ".avi"
        with open(folderpath + 'image_list.txt','w') as f:
            for image_path1 in image_paths:
                f.write(f"file '{os.path.abspath(image_path1)}'\n")
        try:
            ffmpeg.input(folderpath + 'image_list.txt', format='concat', safe=0, r=10).output(output_anime, vcodec='libvpx-vp9',crf=10, pix_fmt='yuv420p',vf=f'{"scale=3840:-1"}', an=None).overwrite_output().run(capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            pprint.pp("FFmpeg error:", e)
            pprint.pp("Standard Output:", e.stdout.decode('utf-8'))
            pprint.pp("Standard Error:", e.stderr.decode('utf-8'))   
        os.remove(folderpath + 'image_list.txt')
        pprint.pp(output_anime)
    
    
    
    
if __name__ == "__main__":
    cms = cfg_my_settings()
    my_colorbar = cms.my_colorbar
    
    font_families = sorted(set(f.name for f in fm.fontManager.ttflist))
    pprint.pp(font_families)

    text1 = r"abcde$^{-1.234} \psi \alpha \Sigma^{\beta\dot\gamma\pm\lambda} \Omega \times \Delta \bullet \zeta \Xi $"
    text2 = r"$\bm abcde^{-1.234} \psi \alpha \Sigma^{\beta\dot\gamma\pm\lambda} \Omega \times \Delta \bullet \zeta \Xi $"
    text_normal = r"abc=defgh+ijklm-nopqr~stuvwx/yz1234|567890"

    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig = plt.figure()
    pgf = plt.get_current_fig_manager()
    #pgf.window.showMaximized()
    ax = plt.subplot(2,2,1)
    cf = plt.contourf(X, Y, Z)
    
    ax.set_xlabel(text_normal)
    ax.set_ylabel(text_normal)
    ax.set_title(text_normal)
    ax.text(-.9, 0.9, text_normal)
    ax.text(-.9, 0.3, text_normal, fontsize=16)
    ax.text(-.9, -0.3, rf"${text_normal}$", fontsize=18)
    ax.text(-.9, -0.6, rf"{text1}\\{text2}", fontsize=18)
    ax.text(.9, 0.9, r"$(a)$")
    
    pos1 = ax.get_position()
    pos2 = Bbox.from_bounds( pos1.x0-0.05 , pos1.y0 ,  0.4 , 0.4 )
    ax.set_position(pos2)
    my_colorbar(cf, ax )
    
    ax = plt.subplot(2,2,3)
    cf = plt.contourf(X, Y, Z)
    ax.set_aspect("equal")
    pos1 = ax.get_position()
    pos2 = Bbox.from_bounds( pos1.x0-0.15 , pos1.y0-0.05 ,  0.3 , 0.3 )
    ax.set_position(pos2)
    my_colorbar(cf, ax , cb_width=0.05)
    
    ax = plt.subplot(2,2,2)
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    cyc = plt.rcParams['axes.prop_cycle']
    cyc = [item['color'].replace('#', r'\#') for item in cyc]
    pprint.pp(cyc)
    for i in range(20,0,-1):
        ax.plot(x, y + i * 0.5, label=f'Line {cyc[20-i]}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('20 Popular Line Colors in Matplotlib')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    pos1 = ax.get_position()
    pos2 = Bbox.from_bounds( pos1.x0+0.04 , pos1.y0+0.05 ,  0.3 , 0.3 )
    ax.set_position(pos2)
    
    plt.show()
    
    
