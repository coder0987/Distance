const fs = require('fs');

const HEIGHT = 28;
const WIDTH = 28;
const ACTIVE = 1;
const SQRT2 = 1.41421356237;

const IMG_COUNT = 1000;
const GOAL_IMG = 240;


//Sum the distances between each active pixel and the closest active pixel on the other image
function distanceStep(imgA, imgB) {
    //Each image is a HEIGHT x WIDTH array of values 0-255
    let activeA = [];
    let activeB = [];

    //Create arrays of active pixels
    for (let r in imgA) {
        for (let c in imgB) {
            if (imgA[r][c] == ACTIVE) {
                activeA.push([r,c]);
            }
            if (imgB[r][c] == ACTIVE) {
                activeB.push([r,c]);
            }
        }
    }

    //For each pixel in A, find the closest active pixel in B and add to the sum
    let sum = 0;
    for (let i in activeA) {
        let cp = 0;
        let closest = distanceTo(activeA[i],activeB[0]);
        for (let j in activeB) {
            let current = distanceTo(activeA[i],activeB[j]);
            if (current < closest) {
                closest = current;
                cp = +j;
            }
        }
        sum += closest;
    }
    //Repeat that process in reverse for symmetry
    for (let i in activeB) {
        let cp = 0;
        let closest = distanceTo(activeB[i],activeA[0]);
        for (let j in activeA) {
            let current = distanceTo(activeB[i],activeA[j]);
            if (current < closest) {
                closest = current;
                cp = +j;
            }
        }
        sum += closest;
    }
    return sum/2;
}

//Iteratively finds the distance between active pixels, then removes pixels that have been used. Unused pixels count as 2
function distanceStepRemoval(imgA, imgB) {
    //Each image is a HEIGHT x WIDTH array of values 0-255
    let activeA = [];
    let activeB = [];
    let pixA = 0;
    let pixB = 0;

    //Create arrays of active pixels
    for (let r in imgA) {
        for (let c in imgB) {
            if (imgA[r][c] == ACTIVE) {
                pixA++;
                activeA.push([r,c]);
            }
            if (imgB[r][c] == ACTIVE) {
                pixB++;
                activeB.push([r,c]);
            }
        }
    }

    //For each pixel in A, find the closest active pixel in B, deactivate that pixel, and add to the sum
    let sum = 0;
    for (let i=0; i<Math.min(pixA, pixB); i++) {
        let cp = 0;
        let closest = distanceTo(activeA[i],activeB[0]);
        for (let j in activeB) {
            let current = distanceTo(activeA[i],activeB[j]);
            if (current < closest) {
                closest = current;
                cp = +j;
            }
        }
        sum += closest;
        activeB.splice(cp, 1);
    }
    return sum + Math.abs(pixA - pixB) * 2;
}

function distanceTo(a, b) {
    return Math.sqrt(Math.pow(+a[0] - +b[0], 2) + Math.pow(+a[1] - +b[1],2));
}

//Treat each pixel as a dimension, then find the magnitude of the vector distance between the images
function euclideanDistance(imgA, imgB) {
    let sum = 0;
    for (let c = 0; c < WIDTH; c++) {
        for (let r = 0; r < HEIGHT; r++) {
            sum += Math.pow(imgA[c][r] - imgB[c][r],2);
        }
    }
    return Math.sqrt(sum);
}

//Sums the active distances of each pixel, then takes the differences
//Downside: cannot detect flips along the top-left to bottom-right diagonal
function pythagoreanDistance(imgA, imgB) {
    //Top left corner
    let aDistance = 0;
    let bDistance = 0;

    for (let c = 0; c < WIDTH; c++) {
        let c2 = c * c;
        for (let r = 0; r < HEIGHT; r++) {
            if (imgA[c][r] == ACTIVE || imgB[c][r] == ACTIVE) {
                //Pixel is active
                let pd = Math.sqrt(c2 + r * r);
                if (imgA[c][r] == ACTIVE) {
                    aDistance += pd;
                }
                if (imgB[c][r] == ACTIVE) {
                    bDistance += pd;
                }
            }
        }
    }
    return Math.abs(aDistance - bDistance);
}

//Weight the height and width, count, and check. Fixes downside of pythagorean distance
//Downside: clumped vs sparse values give similar totals and so have low edit distance
function scaledDimensionalDistance(imgA, imgB) {
    let aDistance = [0,0,0,0];
    let bDistance = [0,0,0,0];

    for (let c = 0; c < WIDTH; c++) {
        let wxw = (WIDTH - c) / WIDTH;
        let xw = c / WIDTH;
        for (let r = 0; r < HEIGHT; r++) {
            if (imgA[c][r] == ACTIVE || imgB[c][r] == ACTIVE) {
                //Pixel is active
                let hyh = (HEIGHT - r) / HEIGHT;
                let yh = r / HEIGHT;
                if (imgA[c][r] == ACTIVE) {
                    aDistance[0] += hyh + wxw;
                    aDistance[1] += hyh + xw;
                    aDistance[2] += yh + wxw;
                    aDistance[3] += yh + xw;
                }
                if (imgB[c][r] == ACTIVE) {
                    bDistance[0] += hyh + wxw;
                    bDistance[1] += hyh + xw;
                    bDistance[2] += yh + wxw;
                    bDistance[3] += yh + xw;
                }
            }
        }
    }

    return Math.sqrt(Math.pow(aDistance[0] - bDistance[0], 2) + Math.pow(aDistance[1] - bDistance[1], 2) + Math.pow(aDistance[2] - bDistance[2], 2) + Math.pow(aDistance[3] - bDistance[3], 2));
}

//Place the image matrix on a 3d sphere and add up the normal vectors, then find the magnitude of the difference between normal vectors
//Downside: sane as SDD, sparse and clumped values give similar totals
function sphericalDistance(imgA, imgB) {
    let aDistance = [0,0,0];
    let bDistance = [0,0,0];
    let pixA = 0;
    let pixB = 0;

    const HORIZONTAL_FACTOR = (WIDTH * SQRT2);
    const VERTICAL_FACTOR = (HEIGHT * SQRT2);

    for (let c = 0; c < WIDTH; c++) {
        let x = c / HORIZONTAL_FACTOR;
        x2 = x * x;
        for (let r = 0; r < HEIGHT; r++) {
            if (imgA[c][r] == ACTIVE || imgB[c][r] == ACTIVE) {
                //Pixel is active
                let y = r / VERTICAL_FACTOR;
                let z = Math.sqrt(1 - y * y - x2);
                if (imgA[c][r] == ACTIVE) {
                    pixA++;
                    aDistance[0] += x;
                    aDistance[1] += y;
                    aDistance[2] += z;
                }
                if (imgB[c][r] == ACTIVE) {
                    pixB++;
                    bDistance[0] += x;
                    bDistance[1] += y;
                    bDistance[2] += z;
                }
            }
        }
    }

    return Math.sqrt(Math.pow(aDistance[0] - bDistance[0], 2) + Math.pow(aDistance[1] - bDistance[1], 2) + Math.pow(aDistance[2] - bDistance[2], 2));
}

function printBinImg(binImg) {
    let str = '';
    for (let i in binImg[0]) {
        for (let j in binImg) {
            str += binImg[j][i] ? 'O' : ' ';
        }
        str += '\n';
    }
    console.log(str);
}

function printSideBySideBinImg(imgA, imgB) {
    let str = '';
    for (let i in imgA[0]) {
        let ts = '';
        for (let j in imgA) {
            str += imgA[j][i] ? 'O' : ' ';
            ts += imgB[j][i] ? 'O' : ' ';
        }
        str += '|' + ts + '\n';
    }
    console.log(str);
}

function compares() {
    printBinImg(arr_imgs[0])
    compDS();
    compDSR();
    compED();
    compPD();
    compSDD();
    compSD();
}
function compDS() {
    console.log("Distance Step");
    let min = 1;
    let minP = distanceStep(arr_imgs[0], arr_imgs[1]);
    for (let i=2; i<count; i++) {
        let d = distanceStep(arr_imgs[0], arr_imgs[i]);
        if (d < minP) {
            min = i;
            minP = d;
        }
    }
    console.log("DS Lowest: " + min);
    console.log("DS Distance: " + minP);
    console.log("Image:");
    printSideBySideBinImg(arr_imgs[0],arr_imgs[min])
}
function compDSR() {
    console.log("Distance Step Removal");
    let min = 1;
    let minP = distanceStepRemoval(arr_imgs[0], arr_imgs[1]);
    for (let i=2; i<count; i++) {
        let d = distanceStepRemoval(arr_imgs[0], arr_imgs[i]);
        if (d < minP) {
            min = i;
            minP = d;
        }
    }
    console.log("DSR Lowest: " + min);
    console.log("DSR Distance: " + minP);
    console.log("Image:");
    printSideBySideBinImg(arr_imgs[0],arr_imgs[min])
}
function compED() {
    console.log("Euclidean Distance");
    let min = 1;
    let minP = euclideanDistance(arr_imgs[0], arr_imgs[1]);
    for (let i=2; i<count; i++) {
        let d = euclideanDistance(arr_imgs[0], arr_imgs[i]);
        if (d < minP) {
            min = i;
            minP = d;
        }
    }
    console.log("ED Lowest: " + min);
    console.log("ED Distance: " + minP);
    console.log("Image:");
    printSideBySideBinImg(arr_imgs[0],arr_imgs[min])
}
function compPD() {
    console.log("Pythagorean Distance");
    let min = 1;
    let minP = pythagoreanDistance(arr_imgs[0], arr_imgs[1]);
    for (let i=2; i<count; i++) {
        let d = pythagoreanDistance(arr_imgs[0], arr_imgs[i]);
        if (d < minP) {
            min = i;
            minP = d;
        }
    }
    console.log("PD Lowest: " + min);
    console.log("PD Distance: " + minP);
    console.log("Image:");
    printSideBySideBinImg(arr_imgs[0],arr_imgs[min])
}
function compSDD() {
    console.log("Scaled Dimensional Distance");
    let min = 1;
    let minP = scaledDimensionalDistance(arr_imgs[0], arr_imgs[1]);
    for (let i=2; i<count; i++) {
        let d = scaledDimensionalDistance(arr_imgs[0], arr_imgs[i]);
        if (d < minP) {
            min = i;
            minP = d;
        }
    }
    console.log("SDD Lowest: " + min);
    console.log("SDD Distance: " + minP);
    console.log("Image:");
    printSideBySideBinImg(arr_imgs[0],arr_imgs[min])
}
function compSD() {
    console.log("Spherical Distance");
    let min = 1;
    let minP = sphericalDistance(arr_imgs[0], arr_imgs[1]);
    for (let i=2; i<count; i++) {
        let d = sphericalDistance(arr_imgs[0], arr_imgs[i]);
        if (d < minP) {
            min = i;
            minP = d;
        }
    }
    console.log("SD Lowest: " + min);
    console.log("SD Distance: " + minP);
    console.log("Image:");
    printSideBySideBinImg(arr_imgs[0],arr_imgs[min])
}

function test() {
    let i1 = [[255,0],[255,0]];
    let i2 = [[0,255],[255,0]];
    let srd = distanceStepRemoval(i1, i2);
    let pd = pythagoreanDistance(i1, i2);
    console.log("Step Removal Distance is " + srd);
    console.log("Pythagorean Distance is " + pd);
}

function printImage(bytes) {
    let printStr = '';
    let r = 0;
    let c = 0;
    for (b in bytes) {
        if (r == 28) {
            if (c == 28) {break;}
            r = 0;
            c++;
            printStr += '\n';
        }
        if (bytes[b]) {
            printStr += 'O';
        } else {
            printStr += ' ';
        }
        r++;
    }
    console.log(printStr);
}

function bufferToImage(bytes) {
    let img = [];
    for (let i=0; i<WIDTH; i++) {img.push([])};
    let r = 0;
    let c = 0;
    for (b in bytes) {
        if (r == 28) {
            if (c == 28) {break;}
            r = 0;
            c++;
        }
        if (bytes[b]) {
            img[r][c] = 1;
        } else {
            img[r][c] = 0;
        }
        r++;
    }
    return img;
}

function mnist(imgNum, base) {
    let data = [];
    let offset = 16 + (28*28*imgNum);
    let i = 0;
    let dataToRead = Math.ceil(28 * 28);
    fs.open('train-images.idx3-ubyte', 'r', function(err, fd) {
        if (err)
          throw err;
        const buffer = Buffer.alloc(1);
        while (i < dataToRead)
        {   
          const num = fs.readSync(fd, buffer, 0, 1, i + offset);
          if (num === 0)
            break;
          data.push(buffer[0]);
          i++;
        }
        assembleAsync(data, base);
    });
}

let count = 1;
let arr_imgs = [];
function assembleAsync(data, base) {
    if (base) {
        arr_imgs[0] = bufferToImage(data);
    } else  {
        arr_imgs[count] = bufferToImage(data);
        count++;
    }
    if (count == IMG_COUNT) {
        compares();
    }
}

mnist(GOAL_IMG, true);


for (let i = 0; i<IMG_COUNT; i++) {
    if (i == GOAL_IMG) {continue;}
    mnist(i, false);
}

