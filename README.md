# learning package

SLA (sequential learning algorithms)

The SLA get past predictions from an ensamble of experts/models and past observations and use that to weight the ensamble member based on their performance

Currently this package contains the Exponential Gradient Average (EGA) algorithm. This algorithm is described and used in the folowing papers:

* Strobach, E., and G. Bel., 2017. “Qunatifying the uncertainties in an ensemble of decadal climate predictions”. J. Geophys. Res., 122, doi:10.1002/2017JD027249.
* Strobach, E. and G. Bel., 2017. “The relative contribution of the internal and model variabilities to the uncertainty in decadal climate predictions”. Climate dynamics, 1-15, doi:10.1007/s00382-016-3507-7.
* Strobach, E., and G. Bel, 2016. “Decadal climate predictions using sequential learning algorithms”. Journal of Climate, 29 (10), 3787–3809, doi:10.1175/JCLI-D-15-0648.1

The Learn Eta Algorithm (LEA) is an experimental modification of the learn-alpha algorithm (LAA; Monteleoni and Jaakkola, 2003). In addition to learning model performance, the LEA algorithm learns the optimal leraning rate.
