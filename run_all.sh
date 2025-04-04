run () {
    cd $1 && echo $1 
    
    echo testing...
    if !( g++ test.cpp -o test -std=c++20 -O2 && ./test 2> test.log && rm ./test ); then
        echo !!!!  tests failed  !!!!
    else
        echo tests passed
    fi

    python3 bench.py 3 26
    python3 plot.py
    
    cd ..
}

for folder in `ls -d */`; do
    if [[ $folder != *"text"* ]]; then
        run $folder
    fi
done
