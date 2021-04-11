REINVENT 3.0 - beta
=======================================================================================

NOTE
-----
The code is still undergoing changes and the provided input examples might not be up-to date yet.
We will try to get this up to date as soon as we can. 


__In the meantime, when in doubt, ask!__

Usage
-----

1. Sample inputs are provided in reinvent/configs/sample_inputs folder.

2. (Recommended) Use jupyter notebooks in `Reinvent Community` repo to generate inputs.

-------------------------------------------------
To use Tensorboard for logging:

   1. To launch Tensorboard, write:
       tensorboard --logdir "path to your log output directory" --port=8008.
       This will give you an address to copy to a browser and access to the graphical summaries from Tensorboard.

   2. Further commands to Tensorboard to change the amount of scalars,histograms, images, distributions and graphs shown
        can be done as follows:
        --samples_per_plugin=scalar=700, images=20

Installation
-----

1. Install Anaconda / Miniconda
2. Clone the repository
3. (Optional) Checkout the appropriate branch of the repository and create a new local branch tied to the remote one, e.g.:
    git checkout --track origin/reinvent.3.0
4. Open terminal, go to the repository and generate the appropriate environment:
    conda env create -f reinvent_shared.yml
   Hint: Use the appropriate `conda` binary. You might want to check, whether you succeeded:
    conda info --envs
5. Since there are components that use OpenEye libraries, if you intend to use them you will need to set the environmental variable OE_LICENSE to activate the oechem license. One way to do this and keep it conda environment specific is:
   On the command line, first:

       cd $CONDA_PREFIX
       mkdir -p ./etc/conda/activate.d
       mkdir -p ./etc/conda/deactivate.d
       touch ./etc/conda/activate.d/env_vars.sh
       touch ./etc/conda/deactivate.d/env_vars.sh

   then edit ./etc/conda/activate.d/env_vars.sh as follows:

       #!/bin/sh
       export OE_LICENSE='<path to OpenEye license file>'

   and finally, edit ./etc/conda/deactivate.d/env_vars.sh :

       #!/bin/sh
       unset OE_LICENSE
6. Activate environment (or set it in your GUI)
7. In the project directory, in ./configs/ create the file `config.json` by copying over `example.config.json` and editing as required


Tests
-----
Currently all tests are excluded form this repository.
