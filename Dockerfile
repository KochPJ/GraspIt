FROM nvcr.io/nvidia/isaac-sim:2023.1.0-hotfix.1

RUN echo 'alias omni_python="/isaac-sim/python.sh"' >> ~/.bashrc

ENTRYPOINT ["sh", "/omniverse/startup.sh"]