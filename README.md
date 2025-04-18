# MacroMicroModeling
building N times particle gradation models and property models
including two main part:

1. building lithological model
read N randomly generated sets of Particle Gradation(PG) data, and interpolate them respectively.
① read each '.txt' file iteratively
② estimate Variogram parameters for each PG endmember and interpolate it
③ abstract lithological model from N sets of PG models (This part is not included in the code)



2.predicting property models
read N sets of PG models and w, e models to precdict each property model
① select one property sample (training) data to be estimated and read it(in '.txt' file form),
② train the BP-ANN model
③ read w,e models and current set of PG model, using the trained BP-ANN model to predict property model
④ iteratively read N sets PG models, we can get N sets of property models

run above code for properties of a1-2,a2-3,Es1-2,...,Ip, respectively, then we can obtain each property's model.
