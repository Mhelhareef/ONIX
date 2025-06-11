import numpy as np
import time
import openmc
from onix.cell import Cell
from onix.system import System
from onix import salameche
from .couple_openmc import *
from .openmc_fix import *
from onix.salameche import mat_builder as mb
from onix.salameche import cram

from onix import utils
from onix import data

class Couple_msr(Couple_openmc):
    """This class is used to execute coupled-mode simulations, for circulating fuel systems

    Through this class, the user can:

    - chose to parallelize the Monte Carlo simulations via the "set_MPI" method

    - set the nuclear data libraries to be used in the simulation

    - set the burnup/time sequence for the simulation
    - define the fuel circulation loop by defining the connetions between OpenMC cells.
    - select which of the Cells defined in the OpenMC input should be depleted (these cells will be known as BUCells)
    - select a list of nuclides in each BUCell which cross sections will be tallied. If the user does not specify any such list of nuclides, ONIX will by
    default tally the cross sections of all
      nuclides which data are found in the cross section library.

    The Couple_openmc class also takes care of the coupling between ONIX and OpenMC. At the beginning of a simulation,
    it imports certain initial parameters from the OpenMC input to ONIX such as the name of the BUCells and the initial
    nuclide densities of each BUCell. During the simulation, it makes sure that information such as the flux tallies,
    neutron spectrum tallies, the reaction rates and the updated nuclide densities are correctly transfered between the burnup
    module and the neutron transport module.

    The Couple_openmc class will also sample the isomeric branching ratios and (n,gamma) cross sections on the
    the same energy points to prepare for the calculations of one-group isomeric branching ratios.

    **IMPORTANT: the root cell in the OpenMC input should be named "root cell"**

    Parameters
    ----------
    MC_input_path : str
        Specifies where the OpenMC input files in .xml format are located. This parameter should not be specified as ONIX cannot yet work without Python API.

    xs_mode : str
        Choice between 'constant lib' and 'no constant lib'. Default to 'no constant lib'.
        'constant lib' indicates to ONIX that the user has provided a constant one-group cross section libraries to be used.
        'no constant lib' indicates to ONIX that it should use one-group cross section computed by OpenMC.

    MPI : str
        Choice between 'on' and 'off'.
        Indicates whether OpenMC will be parrallelized or not.
    
    """




    def __init__(self, MC_input_path = None, xs_mode = 'no constant lib', MPI = None, vol_flow_rate = 0):

        super().__init__(MC_input_path, xs_mode, MPI)

        # By default flow rate in the loop is set to 0 m3/s
        self._vol_flow_rate = vol_flow_rate

    @property
    def vol_flow_rate(self):
        return self._vol_flow_rate

    @vol_flow_rate.setter
    def vol_flow_rate(self, value):
        self._vol_flow_rate = value

    def select_bucells(self, loop_dict, loop_nucl_list=None):
        """Selects the cells loop from the OpenMC input that should be depleted.

        If the user does not specify for which nuclides cross sections should be updated,
        only the OpenMC cell objects should be entered. ONIX will by default calculate the cross sections
        for all nuclides available in the cross section library.

        Parameters
        ----------
        loop_dict: a dicotonray of OpenMC cell names. The value for each key is a dicrotary with two keys
                   the 'feed' key is a list of tuples where the 1st input is an openMC cell names that feed materials to the key cell
                   and the second input is the fraction of the volumetric flow rate comeing from this cell.
                   the 'sink' key is a list of tuples where the 1st input is an openMC cell names that recives materials to the key cell
                   and the second input is the fraction of the volumetric flow rate going into this cell.

        loop_nucl_list: the list of nuclides to be tracked
        """
        self.loop_dict = loop_dict
        self.loop_nucl_list = loop_nucl_list

        self.selected_bucells_name_list = list(self.loop_dict.keys())
        #self.selected_bucells_nucl_list_dict = {key: self.loop_nucl_list for key in self.loop_dict}
        
        

        if self.loop_nucl_list is not None:
            self.selected_bucells_nucl_list_dict = {key: self.loop_nucl_list for key in self.loop_dict}
        else:
            self.selected_bucells_nucl_list_dict = {}


    def burn(self):
        """Launches the coupled simulation.
        """

        start_time = time.time()

        # If no decay libs have been set, set default libs
        if self._decay_lib_set == 'no':
            self.set_default_decay_lib()
            print ('\n\n\n----  Default decay constants library set for system  ----\n---- {} ----'.format(data.default_decay_b_lib_path))
        else:
            print ('\n\n\n----  User defined path for decay library  ----\n\n')
            print ('----  {}  ----\n\n\n'.format(self._decay_lib_path))
        

        # If user has not set a fission yield library, default library (ENDF/B-VII.O) is set
        if self._user_fy_lib_set == 'no':
            self.set_default_fy_lib()
            print ('\n\n\n----  Default fission yields library set for system  ----\n---- {} ----'.format(data.default_fy_lib_path))
        
        # If user has set a fission library, two options:
        elif self._user_fy_lib_set == 'yes':
            print ('\n\n\n----  User defined path for fission yields library ----\n\n')
            print ('----  {}  ----\n\n\n'.format(self._user_fy_lib_path))

            # 1) Default library used to complete user library
            if self._complete_user_fy_lib == 'yes':
                print ('\n\n\n----  User defined fission yields library completed with default fission yields library ----\n\n')
                print ('----  {}  ----\n\n\n'.format(data.default_fy_lib_path))

            # 2) Default library is not used, ONIX only uses user defined fission yields library
        
        #print (self.xs_mode, self._xs_lib_set)
        if self.xs_mode == 'constant lib' and self._xs_lib_set == 'no':
            self.set_default_xs_lib()
            print ('\n\n\n----Default cross section library set for system----\n\n\n')
        else:
            # This method simply passes the MC_XS_nucl_list to each cell so that each cell
            # can then build its own lib_nucl_list
            self._set_MC_XS_nuc_list_to_bucells()

            print ('\n\n\n----  Path for cross sections library ----\n\n')
            print ('----  {}  ----\n\n\n'.format(self._cross_sections_path))

        self._set_sampled_isomeric_branching_data()
        self._set_sampled_ng_cross_section_data()


        system = self.system
        system.zam_order_passlist()
        sequence = system.sequence
        norma_mode = sequence.norma_unit

        bucell_list = system.get_bucell_list()
        dummy_cell = bucell_list[0]
        passlist = dummy_cell.passlist

        m = len(mb.get_initial_vect(passlist))
        n = m * len(bucell_list)
        
        connectivity_matrix = np.zeros((n,n))
        fixed_source = np.zeros(n)

        
        for i, bucell, in enumerate(bucell_list):
            cell_name = bucell.name
            cell_vol = bucell.vol
            passlist = bucell.passlist
            passport_list = passlist.passport_list
            
            if 'upstream_cells' in self.loop_dict[cell_name].keys():
                feed_cell_list = self.loop_dict[cell_name]['upstream_cells']
                sink_flow_fraction = 0
                
                for feed_cell in feed_cell_list:
                    feed_cell_name = feed_cell[0]
                    flow_fraction = feed_cell[1]
                    sink_flow_fraction += flow_fraction
                    feed_rate = self.vol_flow_rate * flow_fraction /cell_vol
                    j = self.selected_bucells_name_list.index(feed_cell_name)
                    connectivity_matrix[i*m:i*m+m, j*m:j*m+m] += feed_rate*np.eye(m)
                

                sink_rate = self.vol_flow_rate * sink_flow_fraction/cell_vol
                connectivity_matrix[i*m:i*m+m, i*m:i*m+m] -= sink_rate*np.eye(m)

            if 'elements_removal' in self.loop_dict[cell_name].keys():
                removed_element = self.loop_dict[cell_name]['elements_removal']
                indices = []
                removal_rates = []
                for pair in removed_element:
                    element = pair[0]
                    rate = pair[1]
                    for index, nucli in enumerate(passport_list):
                        if nucli.name[:2] == element:
                            indices.append(index)
                            removal_rates.append(rate) 
                diagonal_values = [removal_rates[indices.index(k)] if k in indices else 0 for k in range(m)]
                connectivity_matrix[i*m:i*m+m, i*m:i*m+m] -= np.diag(diagonal_values)
                

            if 'nuclide_removal' in self.loop_dict[cell_name].keys():
                removed_nuc = self.loop_dict[cell_name]['nuclide_removal']
                indices = []
                removal_rates = []
                for pair in removed_nuc:
                    nuc = pair[0]
                    rate = pair[1]
                    for index, nucli in enumerate(passport_list):
                        if nucli.name == nuc:
                            indices.append(index)
                            removal_rates.append(rate) 
                diagonal_values = [removal_rates[indices.index(k)] if k in indices else 0 for k in range(m)]
                connectivity_matrix[i*m:i*m+m, i*m:i*m+m] -= np.diag(diagonal_values)

            if 'external_nuclide_addition' in self.loop_dict[cell_name].keys():
                added_nuc = self.loop_dict[cell_name]['external_nuclide_addition']
                indices = []
                addition_rates = []
                for pair in added_element:
                    nuc = pair[0]
                    rate = pair[1]
                    for index, nucli in enumerate(passport_list):
                        if nucli.name == nuc:
                            indices.append(index)
                            addition_rates.append(1e-24*rate/cell_vol) 
                diagonal_values = [addition_rates[indices.index(k)] if k in indices else 0 for k in range(m)]
                fixed_source[i*m:i*m+m] += np.array(diagonal_values)
            
            if 'holdup' in self.loop_dict[cell_name].keys():
                feed_cell_list = self.loop_dict[cell_name]['holdup']
                for feed_cell in feed_cell_list:
                    feed_cell_name = feed_cell[0]
                    removed_elements = feed_cell[1]
                    j = self.selected_bucells_name_list.index(feed_cell_name)
                    feed_cell_volume = bucell_list[j].vol
                    indices = []
                    removal_rates = []
                    for pair in removed_elements:
                        element = pair[0]
                        rate = pair[1]
                        for index, nucli in enumerate(passport_list):
                            if nucli.name[:2] == element:
                               indices.append(index)
                               removal_rates.append(rate) 
                    diagonal_values = [removal_rates[indices.index(k)] if k in indices else 0 for k in range(m)]
                    connectivity_matrix[i*m:i*m+m, j*m:j*m+m] += (feed_cell_volume/cell_vol) * np.diag(diagonal_values)


            if 'external_salt_stream' in self.loop_dict[cell_name].keys():
                stream = self.loop_dict[cell_name]['external_salt_stream']
                flow_rate = stream[0]
                feed_stream = stream[1]
                connectivity_matrix[i*m:i*m+m, i*m:i*m+m] -= flow_rate*np.eye(m)/cell_vol
                indices = []
                density = []
                for pair in feed_stream:
                    nuc = pair[0]
                    feed_density = pair[1]
                    for index, nucli in enumerate(passport_list):
                        if nucli.name == nuc:
                           indices.append(index)
                           density.append(flow_rate*feed_density/cell_vol) 
                diagonal_values = [density[indices.index(k)] if k in indices else 0 for k in range(m)]
                fixed_source[i*m:i*m+m] += np.array(diagonal_values)

            if 'recycling' in self.loop_dict[cell_name].keys():
                recycling = self.loop_dict[cell_name]['recycling']
                destination_cell = recycling[0]
                extracted_nuc = recycling[1]
                j = self.selected_bucells_name_list.index(destination_cell)
                destination_cell_volume = bucell_list[j].vol
                indices = []
                rates = []
                for pair in extracted_nuc:
                    nuc = pair[0]
                    rate = pair[1]
                    for index, nucli in enumerate(passport_list):
                        if nucli.name == nuc:
                           indices.append(index)
                           rates.append(cell_vol*rate/destination_cell_volume)
                diagonal_values = [rates[indices.index(k)] if k in indices else 0 for k in range(m)]
                connectivity_matrix[j*m:j*m+m, i*m:i*m+m] +=  np.diag(diagonal_values)
        
        np.savetxt('connectivity_matrix.csv', connectivity_matrix, delimiter=',')

                
        
        self.connectivity_matrix = connectivity_matrix
        self.fixed_source = fixed_source
        
        for bucell in bucell_list:
            # Check if different nuclide list (initial list, lib list and nucl set (user defined set of nuclides to be considered))
            # are consistent with each other
            bucell._check_nucl_list_consistency()


        #steps_number = sequence.steps_number
        steps_number = sequence.macrosteps_number
        # Shift loop from 1 in order to align loop s and step indexes
        for s in range(1, steps_number+1):

            print ('\n\n\n\n====== STEP {}======\n\n\n\n'.format(s))
            sequence._gen_step_folder(s)
            print (('\n\n\n=== OpenMC Transport {}===\n\n\n'.format(s)))
            self._change_temperature(s)
            self._run_openmc()
            self._set_tallies_to_bucells(s)
            norma = sequence.norma_unit
            if norma == 'power':
                self._step_normalization(s)
            else:
                self._step_normalization2(s)
            self._copy_MC_files(s)
            print (('\n\n\n=== Salameche Burn {} ===\n\n\n'.format(s)))
            print (utils.printer.salameche_header)
            self.burn_step(system, s)
            self._set_dens_to_cells()
        
        # This last openmc_run is used to compute the last burnup/time point kinf
        print ('\n\n\n=== OpenMC Transport for Final Point ===\n\n\n')
        self._run_openmc()

        system._gen_output_summary_folder()
        if self.reac_rank == 'on':
            system._print_summary_allreacs_rank()
        system._print_summary_subdens()
        system._print_summary_dens()
        system._print_summary_xs()
        system._print_summary_flux_spectrum(self.mg_energy)
        system._print_summary_kinf()
        system._print_summary_param()
        system._print_summary_isomeric_branching_ratio()
        self._copy_MC_files(s, final='yes')

        run_time = time.time() - start_time
        print ('\n\n\n >>>>>> ONIX burn took {} seconds <<<<<<< \n\n\n'.format(run_time))


    def burn_step(self, system, s):

        """Depletes the system for macrostep s.

           Parameters
           ----------
           system: onix.System
                  System to be depleted
           s: int
           Macrostep number
        """

        bucell_list = system.get_bucell_list()
        reac_rank = system.reac_rank

        dummy_cell = bucell_list[0]
        passlist = dummy_cell.passlist
        all_nuc = passlist.nucl_list

        m = len(mb.get_initial_vect(passlist))
        n = m * len(bucell_list)

        giant_B = np.zeros((n,n))
        giant_C = np.zeros((n,n))
        giant_N = np.zeros(n)

        connectivity_matrix = self.connectivity_matrix
        fixed_source = self.fixed_source

        for i, bucell in enumerate(bucell_list):
            bucell._set_folder()
            passlist = bucell.passlist
            sequence = bucell.sequence
            
            # microsteps_number is of length s-1
            microsteps_number = sequence.microsteps_number(s-1)
    
            B = mb.get_xs_mat(passlist)
            C = mb.get_decay_mat(passlist)
            N = mb.get_initial_vect(passlist)
            mb._print_all_mat_to_text(B, C, bucell, s)

            giant_B[i*m:i*m+m, i*m:i*m+m]= B
            giant_C[i*m:i*m+m, i*m:i*m+m]= C
            giant_N[i*m:i*m+m]= N


        A = np.zeros((n,n))
        for i in range(microsteps_number):
            for j, bucell in enumerate(bucell_list):
                sequence = bucell.sequence
                bucell_id = bucell.id
                norma = sequence.norma_unit
                sequence._bucell_time_bu_substep_conversion(bucell, s, i)
                
                time_point = sequence.time_subpoint(s, i)
                bucell_bu_point = sequence.bucell_bu_subpoint(s, i)
                time_substep = sequence.get_time_subintvl(s, i)

                pow_dens = sequence.current_pow_dens
                flux = sequence.current_flux

                if i != 0:
                # Check whether actinides are present in the cell
                # If not, then no flux/power update should be done
                    act = bucell.check_act_presence()
                    if act == 'yes':
                # Now that the density of nuclides is updated, calculate the new substep flux or power density
                       if norma == 'power' and flux != 0.0:
                          flux = bucell._update_flux(pow_dens)

                       elif norma == 'flux':
                          pow_dens = bucell._update_pow_dens(flux)

                sequence._set_substep_flux(flux, s, i)
                sequence._set_substep_pow_dens(pow_dens, s, i)


                A[j*m:j*m+m,:] = (giant_B[j*m:j*m+m,:]*1e-24*flux + giant_C[j*m:j*m+m,:] + connectivity_matrix[j*m:j*m+m,:])

            At = A*time_substep
            
            
            print('micro_step:', i)
            giant_N = cram.CRAM16(At, giant_N)
            if np.any(fixed_source != 0):
                print ('Particular solution')
                tt = time.time()
                integral = cram.CRAM16(At, fixed_source) - fixed_source
                Np= np.linalg.solve(A, integral)
                Np[Np < 0] = 0
                print('Particular solution took:{} s'.format(time.time() - tt))
                giant_N += Np
            
            for j, bucell in enumerate(bucell_list):
                

               print(bucell.name)
               cram.CRAM_density_check(bucell, giant_N[j*m:j*m+m])
               bucell._update_dens(giant_N[j*m:j*m+m], i, microsteps_number)

    # Generate the allreacsdic for each nuclide
               if reac_rank == 'on':

                  bucell._set_allreacs_dic(s, i, microsteps_number)

            
        for bucell in bucell_list:
            sequence = bucell.sequence
            bucell._set_step_dens()
            sequence._set_macrostep_bucell_bu()
            bucell._change_isotope_density(s)
            bucell._change_total_density(s)
            bucell._print_substep_dens(s)

        if reac_rank == 'on':
           system._print_current_allreacs_rank()
        system._copy_cell_folders_to_step_folder(s)

    def _step_normalization2(self, s):

        system = self.system
        sequence = self.sequence

        bucell_list = system.get_bucell_list()
        tot_flux = 0
        for bucell in bucell_list:
            bucell_sequence = bucell.sequence
            MC_flux = bucell_sequence.current_MC_flux
            tot_flux  += MC_flux
        for bucell in bucell_list:
            bucell_sequence = bucell.sequence
            MC_flux = bucell_sequence.current_MC_flux
            avg_flux = sequence.norma_vector[s-1]

            flux = (MC_flux/tot_flux)*avg_flux
            pow_dens = bucell._update_pow_dens(flux)
            bucell_sequence._set_macrostep_flux(flux)
            bucell_sequence._set_macrostep_pow_dens(pow_dens)

