# Module containing generic Solver and Restart definitions

from sys import path
path.insert(0, '.')
import pyProblem
import pyVector as Vec
import pySolver
import pyStepper
import atexit
import os
# Functions and modules necessary for writing on disk
import pickle
import re
import numpy as np

import sep_util as sepu
from sys_util import mkdir

from shutil import rmtree
from copy import deepcopy
import datetime


class AugLagrangianSolver:
    """Solver parent object"""

    # Default class methods/functions
    def __init__(self, inner_solver, rho, p_rho, constraint_tol=0.25, m_rho=1, outer=1, save_dual=False):
        """Default class constructor for Solver"""
        self.p_solver = inner_solver
        self.rho = rho
        self.p_rho = p_rho
        self.m_rho = m_rho
        self.c_tol = constraint_tol
        self.outer = outer
        self.save_dual = save_dual
        return

    def run(self, problem, verbose=False, restart=False):
        c_ratio = 0
        dual_count = 0
        inner_count = 0
        start_iter = 0 
        for it in range(self.outer):
            problem.set_rho(self.rho)
            while True:
                problem.setDefaults()
                # temporary solution for resetting the stepper
                problem.stepper = pyStepper.ParabolicStep()

                if verbose:
                    msg = 90 * "*" + "\n"
                    msg += "\t\t\tAUGMENTED LAGRANGIAN (METHOD OF MULTIPLIERS)\n"
                    msg += "\t Rho value used: %.5f\n" % self.rho
                    msg += "\t Inner problem solved %d times\n" % inner_count
                    msg += "\t Dual variable updated %d times\n" % dual_count
                    msg += 90 * "*" + "\n"
                    print(msg)
                    self.p_solver.logger.addToLog(msg)
                
                # Solver inner problem
                self.p_solver.run(problem,verbose,restart)
                inner_count += 1

                # c_ratio = ||mod_res_final||/||max(mod_res)||
                max = np.amax(self.p_solver.obj_terms[:,1])
                if max > 0:
                    c_ratio = self.p_solver.obj_terms[-1,1] / max
                else:
                    c_ratio = 0
                
                if verbose:
                        msg = "\t\t\tCurrent decrease in the constraint-residual norm: %.5f\n" % c_ratio

                if c_ratio <= self.c_tol:
                    if verbose:
                        msg += "\t\t\tUpdating dual variable and decreasing rho = %.5f\n" % self.rho
                        print(msg)
                        self.p_solver.logger.addToLog(msg)

                    # Update dual variable
                    problem.update_dual()
                    dual_count += 1
                    if self.save_dual and self.p_solver.prefix is not None:
                        dual_file = self.p_solver.prefix + "_dual.H"  
                        problem.dual.writeVec(dual_file, mode="a")
                    
                    # decrease rho
                    self.rho *= self.m_rho
                    break
                else:
                    if verbose:
                        msg += "\t\t\tKeeping the dual variable and increasing rho\n"
                        print(msg)
                        self.p_solver.logger.addToLog(msg)
                    # increase rho
                    self.rho *= self.p_rho
                    break
                
                

            
            
            
            

            
