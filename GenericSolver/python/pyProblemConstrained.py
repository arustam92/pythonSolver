# Module containing the definition of abstract inverse problems
import pyVector as pyVec
import pyOperator as pyOp
import pyProblem as P
from math import isnan
import numpy as np

class ProblemAugLagrangian(P.Problem):
    """
       NonLinear inverse Augmented Lagrangian problem of the form
            1/2*|f(m)-d|_2 + dual^T (Am-b) + rho^2/2*|Am - b|_2
    """

    def __init__(self, model, data, op, rho, eq_op, eq_rhs=None, dual_prior=None, grad_mask=None):

        # Setting internal vector
        self.model = model
        self.dmodel = model.clone()
        self.dmodel.zero()
        # Gradient vector
        self.grad = self.dmodel.clone()
        # Copying the pointer to data vector
        self.data = data
        # Setting a right-hand side of the equality constraint
        self.eq_rhs = eq_rhs
        # Checking if space of the prior model is constistent with range of regularization operator
        if self.eq_rhs is not None:
            if not self.eq_rhs.checkSame(eq_op.range):
                raise ValueError("Right-hand side space not consistent with range of equality-constraint operator")
        # Setting non-linear and linearized operators
        if not isinstance(op, pyOp.NonLinearOperator):
            raise TypeError("Not provided a non-linear operator!")
        # Setting non-linear stack of operators
        self.op = pyOp.VstackNonLinearOperator(op, eq_op)
        self.rho = rho  # Regularization weight in Augmented Lagrangian
        # Residual vector (data and model residual vectors)
        self.res = self.op.nl_op.range.clone()
        self.res.zero()
        # Dual variable
        if not dual_prior:
            self.dual = self.op.lin_op.ops[1].range.clone()
            self.dual.zero()
        else:
            self.dual = dual_prior
        # Constraints residuals with added dual variable
        self.g_dual = self.op.lin_op.ops[1].domain.clone()
        # Computing dual residual constant for given dual variable
        self.op.lin_op.ops[1].adjoint(False, self.g_dual, self.dual)
        # Dresidual vector
        self.dres = self.res.clone()
        # Checking if a gradient mask was provided
        self.grad_mask = grad_mask
        if self.grad_mask is not None:
            if not grad_mask.checkSame(model):
                raise ValueError("Mask size not consistent with model vector!")
            self.grad_mask = grad_mask.clone()
        # Setting default variables
        self.setDefaults()
        self.linear = False
        # Objective function terms (useful to analyze each term)
        self.obj_terms = [None, None]
        return

    def __del__(self):
        """Default destructor"""
        return

    def resf(self, model):
        """
        Method to return residual vector r = [r_d; r_eq]:
        r_d = f(m) - d;
        r_eq = Am - b or r_m = g(m) - b
        """
        self.op.nl_op.forward(False, model, self.res)
        # Computing r_d = f(m) - d
        self.res.vecs[0].scaleAdd(self.data, 1., -1.)
        # Computing r_m = Am - b
        if self.eq_rhs is not None:
            self.res.vecs[1].scaleAdd(self.eq_rhs, 1., -1.)
        # Scaling by rho rho*r_m
        self.res.vecs[1].scale(self.rho)
        return self.res

    def gradf(self, model, res):
        """
        Method to return gradient vector
        g = F'r_d + A'(dual + rho*r_eq)
        """
        # Setting model point on which the F is evaluated
        self.op.set_background(model)
        # g = rho*A'r_eq + A'dual
        self.op.lin_op.ops[1].adjoint(False, self.grad, res.vecs[1])
        self.grad.scaleAdd(self.g_dual, self.rho, 1.)
        # g = F'r_d + rho*A'r_eq + A'dual
        self.op.lin_op.ops[0].adjoint(True, self.grad, res.vecs[0])
        # Applying the gradient mask if present
        if self.grad_mask is not None:
            self.grad.multiply(self.grad_mask)
        return self.grad

    def dresf(self, model, dmodel):
        """
        Method to return residual vector
        dres = [F + epsilon * (A or G)]dm
        """
        # Setting model point on which the F is evaluated
        self.op.set_background(model)
        # Computing Ldm = dres_d
        self.op.lin_op.forward(False, dmodel, self.dres)
        # Scaling by epsilon
        self.dres.vecs[1].scale(self.rho)
        return self.dres

    def objf(self, res):
        """
        Method to return objective function value
        1/2|f(m)-d|_2 + dual'(Am-b) +rho^2/2*|Am-b|_2
        """
        # data term
        val = res.vecs[0].norm()
        self.obj_terms[0] = 0.5 * val * val
        # model term
        val = res.vecs[1].norm()
        self.obj_terms[1] = 0.5 * val * val
        obj = self.obj_terms[0] + self.obj_terms[1]
        # dual term
        obj += np.real(self.dual.dot(res.vecs[1]))
        return obj

    def update_dual(self):
        self.dual.scaleAdd(self.res.vecs[1],1.,self.rho)
        # Update A'dual term used in the gradient 
        self.op.lin_op.ops[1].adjoint(False, self.g_dual, self.dual)

    def set_rho(self, rho):
        self.rho = rho