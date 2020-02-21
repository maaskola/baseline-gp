
import math
import sys
import torch as t
import gpytorch
import pandas as pd
# from matplotlib import pyplot as plt
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

def parse_coords(coord_strings, separator='x'):
    def parse_one(x):
        return [float(y) for y in x.split(separator)]
    coords = [parse_one(x) for x in coord_strings]
    coords = t.tensor(coords).float()
    # breakpoint()
    # log(DEBUG, 'coords.shape={}'.format(coords.shape))
    if coords.shape[1] > 2:
        print('warning higher dimensional coordinates')
        coords = coords[:, 0:2]
    return coords

def load(path, transpose=False, coord_separator='x'):
    df = pd.read_csv(path, sep='\t', index_col=0, memory_map=True)
    print('loaded {} count matrix {}'.format(df.shape, path))
    # df = fix_duplicate_rows_and_cols(df)
    df = df.astype('float32')
    if transpose:
        print("transposing count data")
        df = df.transpose()
    spot_names = df.index.tolist()
    gene_names = df.columns.tolist()
    # print("converting to sparse")
    matrix = t.tensor(df.values).float()
    array_coords = parse_coords(spot_names, separator=coord_separator)
    # coords = pd.DataFrame(array_coords, index=spot_names)
    return matrix, array_coords, list(df.columns)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def learn(x, y, training_iterations=500):
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x, y, likelihood)
    model = model.cuda()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = t.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        # breakpoint()
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iterations, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    return model, likelihood

def eval(model, likelihood, x, y):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with t.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x))
        # breakpoint()
        rmse = ((y - pred.sample()).pow(2).mean()).sqrt()
        return rmse

def do_one(train_x, train_y, eval_x, eval_y):
    model, likelihood = learn(train_x, train_y)
    rmse = eval(model, likelihood, eval_x, eval_y)
    return rmse

def do_all(path, drop_percentage=0.2):
    count, coord, gene = load(path, transpose=True)
    count = count.cuda()
    coord = coord.cuda()

    label = t.cuda.FloatTensor(len(count)).uniform_() > drop_percentage
    train_count = count[label == True]
    train_coord = coord[label == True]
    eval_count = count[label == False]
    eval_coord = coord[label == False]

    for i in range(train_count.shape[1]):
        print(gene[i])
        rmse = do_one(train_coord, train_count[:, i], eval_coord, eval_count[:, i])
        print("\t".join(['rmse', gene[i], str(rmse.item())]))

if __name__ == "__main__":
    # print(sys.argv)
    if len(sys.argv) < 2:
        print("""please specify one or more paths to CSV files with count data
must have genes in rows and spots in columns
spot names have to be of the form AxB""")
    for path in sys.argv[1:]:
        do_all(path)
