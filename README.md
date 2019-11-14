# Quaternion Lorentz Transformations

My hobby is to transform physics equations written with tensors into a
quaternion form. This formal exercise must leave results confirmed by
experiment in place. Instead of getting one equation however, the result is
always four equations. These well-formed "extra" equations may provide new
avenues for research. At least that is goal.

In the literature, a number of people have figured out how to represent the
Lorentz group using a complex-valued quaternion triple function. The work with
spinors is like this. Because complex-valued quaternions are **not** a division
algebra, such efforts violate my research effort. In 2010, I figured out a
triple triple product to do boosts. In 2013, I noticed the very same function
could also do rotations easily enough.

In 2019, two people with Ph.D.'s strongly disagreed with my claim that a one
parameter function and thus at most four degrees of freedom could possibly
represent all possible Lorentz transformations which has been long known to
have three degrees of rotation for rotations and three for boosts. Because six
is greater than four, my claim must necessarily be wrong, or so it is
reasonably claimed.

This repo contains two Jupyter notebooks which argues my side of the case. Please
read it and let me know what you think.

As Jupyter Notebook downloads:

[Lorentz boost with just h](docs/Lorentz_boosts_w_just_h.ipynb)

[Replies and responses](docs/replies_and_responses.ipynb)


As PDFs:

[Lorentz boost with just h](docs/Lorentz_boosts_w_just_h.pdf) - 14 pages

[Replies and responses](docs/replies_and_responses.pdf) - 27 pages

Or just clone the repo with: 

```
git clone
https://github.com/dougsweetser/Quaternion_Lorentz_Transformations.git
```

Run the Jupyter notebook on your machine. The QH library has all the needed
quaternion functions. If your computer is not setup to run a Jupyter notebook,
the easiest approach may be to download [anaconda](http://anaconda.com).
