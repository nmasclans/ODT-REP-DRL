import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Latex figures
plt.rc( 'text',       usetex = True )
plt.rc( 'font',       size = 18 )
plt.rc( 'axes',       labelsize = 18)
plt.rc( 'legend',     fontsize = 18)
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{color}')


class ChannelVisualizer():

    def __init__(self, caseN):

        self.caseN = caseN

    #--------------------------------------------------------------------------------------------
    #   Methods:                            ODT vs. DNS
    #--------------------------------------------------------------------------------------------

    def build_u_mean_profile(self, y_odt, y_dns, u_odt, u_rt_odt, u_dns):
        
        filename = f"../../data/{self.caseN}/post/u_mean.jpg"
        print(f"\nMAKING PLOT OF MEAN U PROFILE: ODT vs DNS in {filename}" )

        fig, ax = plt.subplots()

        ax.semilogx(y_dns, u_dns,    'k-',  label=r'DNS')
        ax.semilogx(y_odt, u_odt,    'b--', label=r'ODT (calc. from odt inst. vel.)')
        ax.semilogx(y_odt, u_rt_odt, 'r:',  label=r'ODT (calc. during odt execution)')

        ax.set_xlabel(r'$y^+$')
        ax.set_ylabel(r'$u^+$')
        ax.legend(loc='upper left', frameon=False, fontsize=16)
        #ax.set_ylim([0, 30])
        #ax.set_xlim([1, 1000])

        plt.tight_layout()
        plt.savefig(filename, dpi=600)


    def build_u_rmsf_profile(self, y_odt, y_dns, urmsf_odt, vrmsf_odt, wrmsf_odt, urmsf_dns, vrmsf_dns, wrmsf_dns):

        filename = f"../../data/{self.caseN}/post/u_rmsf.jpg"
        print(f"\nMAKING PLOT OF RMS VEL PROFILES: ODT vs DNS in {filename}" )

        fig, ax = plt.subplots()

        ax.plot(y_odt,  urmsf_odt, 'k-',  label=r'$u_{rmsf}/u_\tau$')
        ax.plot(y_odt,  vrmsf_odt, 'b--', label=r'$v_{rmsf}/u_\tau$')
        ax.plot(y_odt,  wrmsf_odt, 'r:',  label=r'$w_{rmsf}/u_\tau$')

        ax.plot(-y_dns, urmsf_dns, 'k-',  label='')
        ax.plot(-y_dns, vrmsf_dns, 'b--', label='')
        ax.plot(-y_dns, wrmsf_dns, 'r:',  label='')

        ax.plot([0,0], [0,3], '-', linewidth=1, color='gray')
        ax.arrow( 30, 0.2,  50, 0, head_width=0.05, head_length=10, color='gray')
        ax.arrow(-30, 0.2, -50, 0, head_width=0.05, head_length=10, color='gray')
        ax.text(  30, 0.3, "ODT", fontsize=14, color='gray')
        ax.text( -80, 0.3, "DNS", fontsize=14, color='gray')

        ax.set_xlabel(r'$y^+$')
        ax.set_ylabel(r'$u_{i,rmsf}/u_\tau$')
        ax.legend(loc='upper right', frameon=False, fontsize=16)
        #ax.set_xlim([-300, 300])
        ax.set_ylim([0, 3])

        plt.tight_layout()
        plt.savefig(filename, dpi=600)


    def build_runtime_vs_post_statistics(self, yplus_odt, um_odt, vm_odt, wm_odt, urmsf_odt, vrmsf_odt, wrmsf_odt, um_odt_rt, vm_odt_rt, wm_odt_rt, urmsf_odt_rt, vrmsf_odt_rt, wrmsf_odt_rt):

        filename = f"../../data/{self.caseN}/post/vel_stat_runtime_vs_postprocessed.jpg"
        print(f"\nMAKING PLOT OF AVERAGED AND RMSF VEL PROFILES calculated at RUNTIME vs. POST-PROCESSED (ODT) in {filename}")

        fig, ax = plt.subplots(3, figsize=(8,10))

        ms = 3; s = 10
        ax[0].plot(yplus_odt,      um_odt,         'k-',  label=r'$<u>^{+}$ (post)')
        ax[0].plot(yplus_odt[::s], um_odt_rt[::s], 'ko',  label=r'$<u>^{+}$ (runtime)', markersize=ms)
        ax[0].set_xlabel(r'$y^+$')
        ax[0].set_ylabel(r'$u^+$')

        ax[1].plot(yplus_odt,      vm_odt,         'b--', label=r'$<v>^{+}$ (post)')
        ax[1].plot(yplus_odt,      wm_odt,         'r:',  label=r'$<v>^{+}$ (post)')
        ax[1].plot(yplus_odt[::s], vm_odt_rt[::s], 'bo',  label=r'$<w>^{+}$ (runtime)', markersize=ms)
        ax[1].plot(yplus_odt[::s], wm_odt_rt[::s], 'ro',  label=r'$<w>^{+}$ (runtime)', markersize=ms)
        ax[1].set_xlabel(r'$y^+$')
        ax[1].set_ylabel(r'$v^+$, $w^+$')

        ax[2].plot(yplus_odt,      urmsf_odt,      'k-',  label=r'$u_{rmsf}^{+}$ (post)')
        ax[2].plot(yplus_odt,      vrmsf_odt,      'b--', label=r'$v_{rmsf}^{+}$ (post)')
        ax[2].plot(yplus_odt,      wrmsf_odt,      'r:',  label=r'$w_{rmsf}^{+}$ (post)')
        ax[2].plot(yplus_odt[::s], urmsf_odt[::s], 'ko',  label=r'$u_{rmsf}^{+}$ (runtime)', markersize=ms)
        ax[2].plot(yplus_odt[::s], vrmsf_odt[::s], 'bo',  label=r'$v_{rmsf}^{+}$ (runtime)', markersize=ms)
        ax[2].plot(yplus_odt[::s], wrmsf_odt[::s], 'ro',  label=r'$w_{rmsf}^{+}$ (runtime)', markersize=ms)
        ax[2].set_ylabel(r'$u_{rmsf}^{+},v_{rmsf}^{+},w_{rmsf}^{+}$')

        for axis in range(3):
            ax[axis].legend(loc="upper right", fontsize="small")

        plt.tight_layout()
        plt.savefig(filename, dpi=600)


    def build_reynolds_stress_diagonal_profile(self, y_odt, y_dns, Rxx_odt, Ryy_odt, Rzz_odt, Rxx_dns, Ryy_dns, Rzz_dns):

        filename = f"../../data/{self.caseN}/post/reynolds_stress_diagonal.jpg"
        print(f"\nMAKING PLOT OF REYNOLDS STRESSES PROFILES (DIAGONAL): ODT vs DNS in {filename}" )

        fig, ax = plt.subplots()

        ax.plot(y_odt,  Rxx_odt, 'k-',  label=r"$<u'u'>/u_\tau^2$")
        ax.plot(y_odt,  Ryy_odt, 'b--', label=r"$<v'v'>/u_\tau^2$")
        ax.plot(y_odt,  Rzz_odt, 'r:',  label=r"$<w'w'>/u_\tau^2$")

        ax.plot(-y_dns, Rxx_dns, 'k-',  label='')
        ax.plot(-y_dns, Ryy_dns, 'b--', label='')
        ax.plot(-y_dns, Rzz_dns, 'r:',  label='')

        ax.plot([0,0], [-1,8], '-', linewidth=1, color='gray')
        ax.arrow( 30, -0.5,  50, 0, head_width=0.05, head_length=10, color='gray')
        ax.arrow(-30, -0.5, -50, 0, head_width=0.05, head_length=10, color='gray')
        ax.text(  30, -0.4, "ODT", fontsize=14, color='gray')
        ax.text( -80, -0.4, "DNS", fontsize=14, color='gray')

        ax.set_xlabel(r'$y^+$')
        ax.set_ylabel(r"$<u_i'u_i'>/u_\tau^2$")
        ax.legend(loc='upper right', frameon=False, fontsize=16)
        #ax.set_xlim([-300, 300])
        ax.set_ylim([-1, 8])

        plt.tight_layout()
        plt.savefig(filename, dpi=600)


    def build_reynolds_stress_not_diagonal_profile(self, y_odt, y_dns, Rxy_odt, Rxz_odt, Ryz_odt, Rxy_dns, Rxz_dns, Ryz_dns):

        filename = f"../../data/{self.caseN}/post/reynolds_stress_not_diagonal.jpg"
        print(f"\nMAKING PLOT OF REYNOLDS STRESSES PROFILES (NOT-DIAGONAL): ODT vs DNS in {filename}" )

        fig, ax = plt.subplots()

        ax.plot(y_odt,  Rxy_odt, 'k-',  label=r"$<u'v'>/u_\tau^2$")
        ax.plot(y_odt,  Rxz_odt, 'b--', label=r"$<u'w'>/u_\tau^2$")
        ax.plot(y_odt,  Ryz_odt, 'r:',  label=r"$<v'w'>/u_\tau^2$")

        ax.plot(-y_dns, Rxy_dns, 'k-',  label='')
        ax.plot(-y_dns, Rxz_dns, 'b--', label='')
        ax.plot(-y_dns, Ryz_dns, 'r:',  label='')

        ax.plot([0,0], [-1,3], '-', linewidth=1, color='gray')
        ax.arrow( 30, 0.2,  50, 0, head_width=0.05, head_length=10, color='gray')
        ax.arrow(-30, 0.2, -50, 0, head_width=0.05, head_length=10, color='gray')
        ax.text(  30, 0.3, "ODT", fontsize=14, color='gray')
        ax.text( -80, 0.3, "DNS", fontsize=14, color='gray')

        ax.set_xlabel(r'$y^+$')
        ax.set_ylabel(r"$<u_i'u_j'>/u_\tau^2$")
        ax.legend(loc='upper right', frameon=False, fontsize=16)
        #ax.set_xlim([-300, 300])
        ax.set_ylim([-1, 3])

        plt.tight_layout()
        plt.savefig(filename, dpi=600)


    #--------------------------------------------------------------------------------------------
    #   Methods:       ODT profiles convergence for increasing averaging time
    #--------------------------------------------------------------------------------------------

    def build_u_mean_profile_odt_convergence(self, y_odt, y_dns, u_odt_converg, u_dns, averaging_times, y_odt_rt = None, u_odt_rt=None):
        """
        Builds a plot of the u_mean profile of ODT data at several averaging times  
        Mean stream-wise direction (u_mean) is already normalized by u_tau.

        Parameters:
            y_odt (np.array):           y+ coordinates of odt data, 
                                        Shape: (num_points_y_odt)
            y_dns (np.array):           y+ coordinates of dns data, 
                                        Shape: (num_points_y_dns)
            u_odt_converg (np.array):   u+_mean of odt data, along y axis and at several averaging times
                                        Shape: (num_points_y_odt,num_averaging_times)
            u_dns (np.array):           u+_mean of dns data, along y axis at end of the simulation
                                        Shape: (num_points_y_dns)             
            averaging_times (np.array): averaging times at which u_mean is obtained
                                        Shape: (num_averaging_times), column vector
        """
        assert u_odt_converg.shape[0] == len(y_odt)
        assert u_odt_converg.shape[1] == len(averaging_times)

        is_runtime_statistics_calculated = ((y_odt_rt is not None) and (u_odt_rt is not None))
        if is_runtime_statistics_calculated:
            assert u_odt_rt.shape[0] == len(y_odt_rt)

        filename = f"../../data/{self.caseN}/post/u_mean_odt_convergence_postprocess_statistics.jpg"
        print(f"\nMAKING PLOT OF MEAN U PROFILE CONVERGENCE of ODT with POST-PROCESSING CALCULATED STATISTICS in {filename}" )

        fig, ax = plt.subplots(figsize=(8,6))
        ax.semilogx(y_odt, u_odt_converg, label = [r"$T_{{avg}}={}$".format(t) for t in averaging_times])
        ax.semilogx(y_dns, u_dns, 'k--', label=r"DNS")
        ax.set_xlabel(r'$y^+$')
        ax.set_ylabel(r'$u^+$')
        ax.legend(loc='upper center', ncol = 3, bbox_to_anchor=(0.5,1.35))
        fig.subplots_adjust(top=0.75, bottom=0.15)  # Leave space for the legend above the first subplot
        plt.savefig(filename, dpi=600)

        if is_runtime_statistics_calculated:

            filename = f"../../data/{self.caseN}/post/u_mean_odt_convergence_runtime_statistics.jpg"
            print(f"\nMAKING PLOT OF MEAN U PROFILE CONVERGENCE of ODT with RUNTIME-CALCULATED STATISTICS in {filename}" )
            
            fig, ax = plt.subplots(figsize=(8,6))
            ax.semilogx(y_odt_rt, u_odt_rt, label = [r"$T_{{avg}}={}$ (rt)".format(t) for t in averaging_times])
            ax.semilogx(y_dns, u_dns, 'k--', label=r"DNS")
            ax.set_xlabel(r'$y^+$')
            ax.set_ylabel(r'$u^+$')
            ax.legend(loc='upper center', ncol = 3, bbox_to_anchor=(0.5,1.35))
            fig.subplots_adjust(top=0.75, bottom=0.15)  # Leave space for the legend above the first subplot
            plt.savefig(filename, dpi=600)


    def build_u_rmsf_profile_odt_convergence(self, y_odt, y_dns, \
                                             urmsf_odt_convergence, vrmsf_odt_convergence, wrmsf_odt_convergence, \
                                             urmsf_dns, vrmsf_dns, wrmsf_dns, averaging_times):

        filename = f"../../data/{self.caseN}/post/u_rmsf_odt_convergence.jpg"
        print(f"\nMAKING PLOT OF RMS VEL PROFILES: ODT vs DNS in {filename}" )

        fig, ax = plt.subplots(3, figsize=(9,9))

        ax[0].plot(y_odt,  urmsf_odt_convergence)
        ax[1].plot(y_odt,  vrmsf_odt_convergence)
        ax[2].plot(y_odt,  wrmsf_odt_convergence)

        ax[0].plot(y_dns, urmsf_dns, 'k--')
        ax[1].plot(y_dns, vrmsf_dns, 'k--')
        ax[2].plot(y_dns, wrmsf_dns, 'k--')
        
        # Axis: labels and limits
        ylabel_str = [r'$u_{rmsf}/u_\tau$', r'$v_{rmsf}/u_\tau$', r'$w_{rmsf}/u_\tau$']
        for axis in range(3):
            ax[axis].set_xlabel(r'$y^+$')
            ax[axis].set_ylabel(ylabel_str[axis])
            ax[axis].set_xlim([0, np.max(np.concatenate([y_odt,y_dns]))])
            ax[axis].set_ylim([0, int(np.max([urmsf_dns,vrmsf_dns,wrmsf_dns])+1)])
        
        # Legend
        # Specify the legend of only for first subplot, idem for other
        labels_averaging_times = [r"$T_{{avg}}={}$".format(t) for t in averaging_times]
        labels_str = labels_averaging_times + ["DNS",]
        ax[0].legend(labels_str, loc='upper center', ncol = 3, bbox_to_anchor=(0.5,1.6))
        fig.subplots_adjust(top=0.85)  # Leave space for the legend above the first subplot
        
        plt.savefig(filename, dpi=600)

    
    def build_reynolds_stress_diagonal_profile_odt_convergence(self, y_odt, y_dns, \
            Rxx_odt_convergence, Ryy_odt_convergence, Rzz_odt_convergence, \
            Rxx_dns, Ryy_dns, Rzz_dns, averaging_times):

        filename = f"../../data/{self.caseN}/post/reynolds_stress_diagonal_odt_convergence.jpg"
        print(f"\nMAKING PLOT OF REYNOLDS STRESSES PROFILES (DIAGONAL): ODT vs DNS in {filename}" )

        fig, ax = plt.subplots(3, figsize=(9,9))

        ax[0].plot(y_odt, Rxx_odt_convergence)
        ax[1].plot(y_odt, Ryy_odt_convergence)
        ax[2].plot(y_odt, Rzz_odt_convergence)

        ax[0].plot(y_dns, Rxx_dns, 'k--')
        ax[1].plot(y_dns, Ryy_dns, 'k--')
        ax[2].plot(y_dns, Rzz_dns, 'k--')

        # Axis: labels and limits
        ylabel_str = [r"$<u'u'>/u_\tau^2$", r"$<v'v'>/u_\tau^2$", r"$<w'w'>/u_\tau^2$"]
        for axis in range(3):
            ax[axis].set_xlabel(r'$y^+$')
            ax[axis].set_ylabel(ylabel_str[axis])
            ax[axis].set_xlim([0, np.max(np.concatenate([y_odt,y_dns]))])
            ax[axis].set_ylim([int(np.min([Rxx_dns,Ryy_dns,Rzz_dns]))-1, int(np.max([Rxx_dns,Ryy_dns,Rzz_dns]))+1])

        # Legend
        # Specify the legend of only for first subplot, idem for other
        labels_averaging_times = [r"$T_{{avg}}={}$".format(t) for t in averaging_times]
        labels_str = labels_averaging_times + ["DNS",]
        ax[0].legend(labels_str, loc='upper center', ncol = 3, bbox_to_anchor=(0.5,1.6))
        fig.subplots_adjust(top=0.85)  # Leave space for the legend above the first subplot

        plt.savefig(filename, dpi=600)


    def build_reynolds_stress_not_diagonal_profile_odt_convergence(self, y_odt, y_dns, \
            Rxy_odt_convergence, Rxz_odt_convergence, Ryz_odt_convergence, \
            Rxy_dns, Rxz_dns, Ryz_dns, averaging_times):

        filename = f"../../data/{self.caseN}/post/reynolds_stress_not_diagonal_odt_convergence.jpg"
        print(f"\nMAKING PLOT OF REYNOLDS STRESSES PROFILES (NOT-DIAGONAL): ODT vs DNS in {filename}" )

        fig, ax = plt.subplots(3, figsize=(9,9))

        ax[0].plot(y_odt, Rxy_odt_convergence)
        ax[1].plot(y_odt, Rxz_odt_convergence)
        ax[2].plot(y_odt, Ryz_odt_convergence)

        ax[0].plot(y_dns, Rxy_dns, 'k--')
        ax[1].plot(y_dns, Rxz_dns, 'k--')
        ax[2].plot(y_dns, Ryz_dns, 'k--')

        # Axis: labels and limits
        ylabel_str = [r"$<u'v'>/u_\tau^2$", r"$<u'w'>/u_\tau^2$", r"$<v'w'>/u_\tau^2$"]
        for axis in range(3):
            ax[axis].set_xlabel(r'$y^+$')
            ax[axis].set_ylabel(ylabel_str[axis])
            ax[axis].set_xlim([0, np.max(np.concatenate([y_odt,y_dns]))])
            ax[axis].set_ylim([int(np.min([Rxy_dns,Rxz_dns,Ryz_dns]))-1, int(np.max([Rxy_dns,Rxz_dns,Ryz_dns]))+1])

        # Legend
        # Specify the legend of only for first subplot, idem for other
        labels_averaging_times = [r"$T_{{avg}}={}$".format(t) for t in averaging_times]
        labels_str = labels_averaging_times + ["DNS",]
        ax[0].legend(labels_str, loc='upper center', ncol = 3, bbox_to_anchor=(0.5,1.6))
        fig.subplots_adjust(top=0.85)  # Leave space for the legend above the first subplot

        plt.savefig(filename, dpi=600)


    def build_stress_decomposition(self, ydelta_odt, ydelta_dns, \
                                   tau_viscous_odt, tau_reynolds_odt, tau_total_odt, \
                                   tau_viscous_dns, tau_reynolds_dns, tau_total_dns):
        
        filename = f"../../data/{self.caseN}/post/stress_decomposition.jpg"
        print(f"\nMAKING PLOT OF STRESS DECOMPOSITION ODT vs DNS in {filename}")

        fig, ax = plt.subplots(2,figsize=(9,9))
        ax[0].set_title("ODT")
        ax[0].plot(ydelta_odt[:-1], tau_viscous_odt[:-1],  'k-',  label=r"$\tau_{viscous}=\rho\nu\,d<U>/dy$"  )
        ax[0].plot(ydelta_odt[:-1], tau_reynolds_odt[:-1], 'b--', label=r"$\tau_{reynolds,uv}=-\rho<u'v'>$"  )
        ax[0].plot(ydelta_odt[:-1], tau_total_odt[:-1],    'r:',  label=r"$\tau_{total}$"  )
        ax[1].set_title("DNS")
        ax[1].plot(ydelta_dns[:-1], tau_viscous_dns[:-1],  'k-',  label=r"$\tau_{viscous}=\rho\nu\,d<U>/dy$"  )
        ax[1].plot(ydelta_dns[:-1], tau_reynolds_dns[:-1], 'b--', label=r"$\tau_{reynolds,uv}=-\rho<u'v'>$"  )
        ax[1].plot(ydelta_dns[:-1], tau_total_dns[:-1],    'r:',  label=r"$\tau_{total}$"  )
        for i in range(2):
            ax[i].set_xlabel(r"$y/\delta$")
            ax[i].set_ylabel(r"$\tau(y)$")
            ax[i].legend(loc='upper right', ncol = 1)
        fig.tight_layout()
        plt.savefig(filename, dpi=600)


    def build_TKE_budgets(self, yplus_odt, yplus_dns, vt_u_plus_odt, d_u_plus_odt, vt_u_plus_dns, p_u_plus_dns):

        filename = f"../../data/{self.caseN}/post/TKE_budgets.jpg"
        print(f"\nMAKING PLOT OF TKE BUDGETS ODT vs DNS in {filename}")
       
        fig, ax = plt.subplots()

        # ODT
        ax.plot(yplus_odt[1:-1], vt_u_plus_odt[1:-1],  '-', color = '#ff7575', label=r'$vt_{u}^{+}$')
        ax.plot(yplus_odt[1:-1], d_u_plus_odt[1:-1],   '-', color = '#22c7c7', label=r'$-d_{u}^{+}$')
        # DNS
        ax.plot(-yplus_dns[1:-1], vt_u_plus_dns[1:-1], '-', color = '#ff7575', label='')
        ax.plot(-yplus_dns[1:-1], p_u_plus_dns[1:-1],  '-', color = '#0505ff', label='')

        arrOffset  = -500 # arrows offset
        textOffset = 100
        ax.plot([0,0], [0,3], '-', linewidth=1, color='gray')
        ax.arrow( 30, arrOffset,  20, 0, head_width=50, head_length=5, color='gray')
        ax.arrow(-30, arrOffset, -20, 0, head_width=50, head_length=5, color='gray')
        ax.text(  30, arrOffset + textOffset, "ODT", fontsize=14, color='gray')
        ax.text( -45, arrOffset + textOffset, "DNS", fontsize=14, color='gray')
        ax.set_xlabel(r'$y^+$')
        ax.set_ylabel("TKE budgets")
        ax.legend(loc='upper right', frameon=False, fontsize=16)
        ax.set_xlim([-100, 100])

        plt.tight_layout()
        plt.savefig(filename, dpi=600)

    
    def build_um_profile_symmetric_vs_nonsymmetric(self, CI, yuplus, um_nonsym, um_sym):

        filename = f"../../data/{self.caseN}/post/u_mean_symmetric_vs_nonsymmetric.jpg"
        print(f"\nMAKING PLOT OF UM+ ORIGINAL NON-SYMMETRIC PROFILE vs SYMMETRIC PROFILE in {filename}")

        fig, ax = plt.subplots()
        ax.plot(yuplus, um_nonsym,  'k-',  label=r"um+ non-sym (original)")
        ax.plot(yuplus, um_sym,     'b--', label=r"um+ symmetric")

        ax.set_xlabel(r'$y^+$')
        ax.set_ylabel(r"$um^+$")
        ax.set_title(f"um+ original (non-sym) vs. symmetric \nCI = {CI:.3f}")

        ax.legend(loc='best', frameon=False, fontsize=16)

        plt.tight_layout()
        plt.savefig(filename, dpi=600)


    def build_um_profile_symmetric_vs_nonsymmetric_odt_convergence(self, CI, yuplus, um_nonsym, um_sym, averaging_times):

        filename = f"../../data/{self.caseN}/post/u_mean_symmetric_vs_nonsymmetric_odt_convergence.jpg"
        print(f"\nMAKING PLOT OF UM+ ORIGINAL NON-SYMMETRIC PROFILE vs SYMMETRIC PROFILE for ODT CONVERGENCE in {filename}")

        num_profiles = um_nonsym.shape[1]

        fig, ax = plt.subplots()
        for p in range(num_profiles):
            ax.plot(yuplus, um_nonsym[:,p], '--', label=f"um+ non-sym:  t = {averaging_times[p]:.0f}, CI = {CI[p]:.1f}")
            ax.plot(yuplus, um_sym[:,p],    ':',  label=f"um+ symmetric: t = {averaging_times[p]:.0f}")

        ax.set_xlabel(r'$y^+$')
        ax.set_ylabel(r"$um^+$")
        ax.set_title(f"um+ original (non-sym) vs. symmetric")

        ax.legend(loc='best', frameon=False, fontsize=16)

        plt.tight_layout()
        plt.savefig(filename, dpi=600)

    def build_CI_evolution(self, time, CI):
        # todo: include time data in the x axis, by now it is just index position in the CI list

        filename = f"../../data/{self.caseN}/post/CI_vs_time.jpg"
        print(f"\nMAKING PLOT OF CI EVOLUTION ALONG TIME in {filename}")

        fig, ax = plt.subplots()
        ax.plot(time, CI)
        ax.set_xlabel("Time (t) [s]")
        ax.set_ylabel("Convergence Indicator (CI)")
        ax.set_ylim([0, 3])

        plt.grid()
        plt.tight_layout()
        plt.savefig(filename, dpi=600)


