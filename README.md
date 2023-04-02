# Neural Network from scratch in golang.

Taken the neural code from the blog:

https://sausheong.github.io/posts/how-to-build-a-simple-artificial-neural-network-with-go/

### Before it all

Unzip the data set in the mnist_dataset.
Inside the zip folder we have some samples of tests
and you dont need to select the csv to make the code run, you can change the csv in the code.

### Run it:

first you will need to clone the repo and enter the folder:

$ git clone https://github.com/mahauni/neural-nerwork-go

$ cd neural-network-go

and then:

$ go build

to train the neural network
$ ./neural-network-go.exe -mnist train

mass predict a bunch of tests
$ ./neural-network-go.exe -mnist predict 


and will add more features to it:

- make a way to draw with it and try the neural network to guess.
