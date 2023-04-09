import gensim.downloader as api
print("success!")

##try:
##    print("Trying to import Gensim")
##    import gensim.downloader as api
##except ImportError:
##    print("Error")
##    try:
##        print("Trying to install Gensim")
##        import subprocess
##        import sys
##        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gensim'])
##        import gensim.downloader as api
##    except subprocess.CalledProcessError:
##        print("Unable to install gensim library")
##
