import numpy as np
import scipy
import scipy.linalg

from autodp.mechanism_zoo import GaussianMechanism
from autodp.transformer_zoo import ComposeGaussian
from autodp.autodp_core import Mechanism
from autodp.calibrator_zoo import eps_delta_calibrator

from sklearn import preprocessing


class GB_mech(Mechanism):
    def __init__(self,sigma,coeff,name='GB'):
        Mechanism.__init__(self)
        self.name = name
        self.params={'sigma':sigma,'coeff':coeff}
        gm = GaussianMechanism(sigma,name='Release')
        # compose them with the transformation: ComposeGaussian.
        compose = ComposeGaussian() 
        mech = compose([gm], [coeff])
        
        self.set_all_representation(mech)

class BoostedAdaSSP:
    def __init__(
        self,
        x_bound=1,
        y_bound=1,
        epsilon=1,
        delta=1e-6,
        num_iterations=100,
        shrinkage="constant",
        random_state=np.random.RandomState(42),
    ):
        self.rng = random_state
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.epsilon = epsilon
        self.delta = delta
        self.num_iterations = num_iterations
        # print(shrinkage)
        if shrinkage == "constant":
            self.shrinkage = lambda x: 1
        if shrinkage == "1/T":
            self.shrinkage = lambda x: 1/x
        if shrinkage == "1/T**0.5":
            self.shrinkage = lambda x: 1/x ** 0.5
       

        self.calibration()

    def calibration(self):
        GB_fix_iterations = lambda x:  GB_mech(x,1+self.num_iterations+1)
        calibrate = eps_delta_calibrator()
        mech = calibrate(GB_fix_iterations, self.epsilon, self.delta, [0,100000])
        self.sigma = mech.params['sigma']


    def clipping_norm(self, X):
        normalized_X = preprocessing.normalize(X, norm="l2")
        length_X = np.linalg.norm(X, axis=1, keepdims=True)
        clipped_X = normalized_X * length_X.clip(min=0, max=self.x_bound)

        return clipped_X

    def noisy_cov(self, XTX):
        # GM1
        Z = self.x_bound**2 * self.sigma * self.rng.normal(size=XTX.shape)

        Z_analyzegauss = np.triu(Z) + np.triu(Z, k=1).T
        hatXTX = XTX + Z_analyzegauss
        # GM3
        s = scipy.linalg.eigvalsh(XTX, subset_by_value=(0, np.inf))
        s = s[::-1]

        lambdamin = s[-1] + self.x_bound**2 * self.sigma * self.rng.normal(size=1)
        lambdamin_lowerbound = max(0, lambdamin - self.x_bound**2 * self.sigma * 1.96)

        dim = XTX.shape[0]
        lamb = max(
            0,
            np.sqrt(dim) * self.sigma * self.x_bound**2 * 1.96 - lambdamin_lowerbound,
        )

        return hatXTX + lamb * np.eye(dim)

    def run_AdaSSP(self, hatXTX, XTy):
        # GM2
        hatXTy = XTy + self.sigma * self.x_bound * self.y_bound * self.rng.normal(
            size=XTy.shape
        )
        theta_adassp = scipy.linalg.solve(hatXTX, hatXTy, assume_a="sym")
        return theta_adassp

    def fit(self, X, y):
        X = self.clipping_norm(X)

        n, dim = X.shape

        XTX = X.T @ X 

        hatXTX = self.noisy_cov(XTX)

        ensemble_theta = np.zeros(dim)

        for i in range(self.num_iterations):
            residual = y - X @ ensemble_theta
            residual = residual.clip(-self.y_bound, self.y_bound)
            XTy = X.T @ residual 

            theta = self.run_AdaSSP(
                hatXTX,
                XTy,
            )

            shrinkage = self.shrinkage((i+1))
            ensemble_theta += shrinkage * theta

        self.ensemble_theta = ensemble_theta
        return self

    def predict(self, X):
        X = self.clipping_norm(X)
        return X @ self.ensemble_theta
