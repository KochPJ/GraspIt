for i in {1..100}; do
  LD_PRELOAD=$HOME/oldlibssl/lib/libssl.so.1.1:$HOME/oldlibssl/lib/libcrypto.so.1.1 /bin/python3 usd2obj.py
done
