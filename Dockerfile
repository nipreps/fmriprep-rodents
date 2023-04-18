# fMRIPrep-rodents Docker Container Image distribution
#
# MIT License
#
# Copyright (c) 2021 The NiPreps Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Use Ubuntu 20.04 LTS
FROM nipreps/miniconda:py39_2205.0

ENV DEBIAN_FRONTEND=noninteractive

# Installing freesurfer
COPY docker/files/freesurfer-exclude.txt /usr/local/etc/freesurfer-exclude.txt
RUN curl -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.1/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1.tar.gz \
    | tar zxv --no-same-owner -C /opt --exclude-from=/usr/local/etc/freesurfer-exclude.txt

# Simulate SetUpFreeSurfer.sh
ENV FSL_DIR="/opt/fsl-5.0.11" \
    OS="Linux" \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT="nii.gz" \
    FREESURFER_HOME="/opt/freesurfer"
ENV SUBJECTS_DIR="$FREESURFER_HOME/subjects" \
    FUNCTIONALS_DIR="$FREESURFER_HOME/sessions" \
    MNI_DIR="$FREESURFER_HOME/mni" \
    LOCAL_DIR="$FREESURFER_HOME/local" \
    MINC_BIN_DIR="$FREESURFER_HOME/mni/bin" \
    MINC_LIB_DIR="$FREESURFER_HOME/mni/lib" \
    MNI_DATAPATH="$FREESURFER_HOME/mni/data"
ENV PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    MNI_PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    PATH="$FREESURFER_HOME/bin:$FSFAST_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH"

# FSL 5.0.11 (neurodocker build)
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           bc \
           dc \
           file \
           libfontconfig1 \
           libfreetype6 \
           libgl1-mesa-dev \
           libgl1-mesa-dri \
           libglu1-mesa-dev \
           libgomp1 \
           libice6 \
           libxcursor1 \
           libxft2 \
           libxinerama1 \
           libxrandr2 \
           libxrender1 \
           libxt6 \
           sudo \
           wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Downloading FSL ..." \
    && mkdir -p /opt/fsl-5.0.11 \
    && curl -fsSL --retry 5 https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-5.0.11-centos6_64.tar.gz \
    | tar -xz -C /opt/fsl-5.0.11 --strip-components 1 \
    --exclude "fsl/config" \
    --exclude "fsl/data/atlases" \
    --exclude "fsl/data/first" \
    --exclude "fsl/data/mist" \
    --exclude "fsl/data/possum" \
    --exclude "fsl/data/standard/bianca" \
    --exclude "fsl/data/standard/tissuepriors" \
    --exclude "fsl/doc" \
    --exclude "fsl/etc/default_flobs.flobs" \
    --exclude "fsl/etc/fslconf" \
    --exclude "fsl/etc/js" \
    --exclude "fsl/etc/luts" \
    --exclude "fsl/etc/matlab" \
    --exclude "fsl/extras" \
    --exclude "fsl/include" \
    --exclude "fsl/python" \
    --exclude "fsl/refdoc" \
    --exclude "fsl/src" \
    --exclude "fsl/tcl" \
    --exclude "fsl/bin/FSLeyes" \
    && find /opt/fsl-5.0.11/bin -type f -not \( \
        -name "applywarp" -or \
        -name "bet" -or \
        -name "bet2" -or \
        -name "convert_xfm" -or \
        -name "fast" -or \
        -name "flirt" -or \
        -name "fsl_regfilt" -or \
        -name "fslhd" -or \
        -name "fslinfo" -or \
        -name "fslmaths" -or \
        -name "fslmerge" -or \
        -name "fslroi" -or \
        -name "fslsplit" -or \
        -name "fslstats" -or \
        -name "imtest" -or \
        -name "mcflirt" -or \
        -name "melodic" -or \
        -name "prelude" -or \
        -name "remove_ext" -or \
        -name "susan" -or \
        -name "topup" -or \
        -name "zeropad" \) -delete \
    && find /opt/fsl-5.0.11/data/standard -type f -not -name "MNI152_T1_2mm_brain.nii.gz" -delete
ENV FSLDIR="/opt/fsl-5.0.11" \
    PATH="/opt/fsl-5.0.11/bin:$PATH" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q" \
    LD_LIBRARY_PATH="/opt/fsl-5.0.11/lib:$LD_LIBRARY_PATH"

# Install AFNI latest (neurodocker build)
ENV AFNI_DIR="/opt/afni"
RUN echo "Downloading AFNI ..." \
    && mkdir -p ${AFNI_DIR} \
    && curl -fsSL --retry 5 https://afni.nimh.nih.gov/pub/dist/tgz/linux_openmp_64.tgz \
    | tar -xz -C ${AFNI_DIR} --strip-components 1
ENV PATH="${AFNI_DIR}:$PATH" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_MODELPATH="${AFNI_DIR}/models" \
    AFNI_TTATLAS_DATASET="${AFNI_DIR}/atlases" \
    AFNI_PLUGINPATH="${AFNI_DIR}/plugins"

# Install AFNI's dependencies
RUN ${CONDA_PATH}/bin/mamba install -c conda-forge -c anaconda \
                            gsl                                \
                            xorg-libxp                         \
                            scipy=1.8                          \
    && ${CONDA_PATH}/bin/mamba install -c sssdgc png \
    && sync \
    && ${CONDA_PATH}/bin/conda clean -afy; sync \
    && rm -rf ~/.conda ~/.cache/pip/*; sync \
    && ln -s ${CONDA_PATH}/lib/libgsl.so.25 /usr/lib/x86_64-linux-gnu/libgsl.so.19 \
    && ln -s ${CONDA_PATH}/lib/libgsl.so.25 /usr/lib/x86_64-linux-gnu/libgsl.so.0 \
    && ldconfig

RUN apt-get update \
 && apt-get install -y -q --no-install-recommends     \
                    libcurl4-openssl-dev              \
                    libgdal-dev                       \
                    libgfortran-8-dev                 \
                    libgfortran4                      \
                    libglw1-mesa                      \
                    libgomp1                          \
                    libjpeg62                         \
                    libnode-dev                       \
                    libssl-dev                        \
                    libudunits2-dev                   \
                    libxm4                            \
                    libxml2-dev                       \
                    netbase                           \
                    netpbm                            \
                    tcsh                              \
                    xfonts-base                       \
 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
 && ldconfig

# Installing ANTs 2.3.3 (NeuroDocker build)
# Note: the URL says 2.3.4 but it is actually 2.3.3
ENV ANTSPATH="/opt/ants" \
    PATH="/opt/ants:$PATH"
WORKDIR $ANTSPATH
RUN curl -sSL "https://dl.dropbox.com/s/gwf51ykkk5bifyj/ants-Linux-centos6_x86_64-v2.3.4.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1

# Installing and setting up ICA_AROMA
WORKDIR /opt/ICA-AROMA
RUN curl -sSL "https://github.com/oesteban/ICA-AROMA/archive/v0.4.5.tar.gz" \
  | tar -xzC /opt/ICA-AROMA --strip-components 1 && \
  chmod +x /opt/ICA-AROMA/ICA_AROMA.py
ENV PATH="/opt/ICA-AROMA:$PATH" \
    AROMA_VERSION="0.4.5"

WORKDIR /opt
COPY --from=nipreps/miniconda@sha256:4d0dc0fabb794e9fe22ee468ae5f86c2c8c2b4cd9d7b7fdf0c134d9e13838729 /opt/conda /opt/conda

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Set CPATH for packages relying on compiled libs (e.g. indexed_gzip)
ENV PATH="/opt/conda/bin:$PATH" \
    CPATH="/opt/conda/include:$CPATH" \
    LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1

RUN conda install -y -n base \
    -c anaconda \
    -c conda-forge \
    convert3d=1.3.0 \
    && sync \
    && conda clean -afy; sync \
    && rm -rf ~/.conda ~/.cache/pip/*; sync \
    && ldconfig

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users fmriprep
WORKDIR /home/fmriprep
ENV HOME="/home/fmriprep"

RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> $HOME/.bashrc && \
    echo "conda activate base" >> $HOME/.bashrc

# Precaching atlases
RUN python -c "import templateflow; \
               templateflow.update(); \
               templateflow.api.get('Fischer344', extension=['.nii', '.nii.gz'])" && \
    find $HOME/.cache/templateflow -type d -exec chmod go=u {} + && \
    find $HOME/.cache/templateflow -type f -exec chmod go=u {} +

COPY . /src/fmriprep
ARG VERSION
# Force static versioning within container
RUN echo "${VERSION}" > /src/fmriprep/fprodents/VERSION && \
    echo "include fprodents/VERSION" >> /src/fmriprep/MANIFEST.in && \
    /opt/conda/bin/python -m pip install --no-cache-dir "/src/fmriprep[all]"

RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} + && \
    rm -rf $HOME/.npm $HOME/.conda $HOME/.empty

ENV IS_DOCKER_8395080871=1

RUN ldconfig
WORKDIR /tmp
ENTRYPOINT ["/opt/conda/bin/fprodents"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="fMRIPrep-rodents" \
      org.label-schema.description="fMRIPrep-rodents - robust fMRI preprocessing tool" \
      org.label-schema.url="http://fmriprep.org" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/nipreps/fmriprep-rodents" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
