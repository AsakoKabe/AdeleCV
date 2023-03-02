
// window.addEventListener("load", function () {
//
// });
// function f () {
//         var elements = document.getElementsByClassName("show-hide");
//         console.log(elements);
//         console.log(elements[0]);
//         elements[0].classList.add("btn");
//         elements[0].classList.add("btn-primary");
//     // let btn = document.getElementsByClassName('show-hide')
//     // console.log(btn)
//     // let b = btn[0]
//     // console.log(b)
// };
// setTimeout(f, 1000);
// document.addEventListener("DOMContentLoaded", function(){
//         var elements = document.getElementsByClassName("show-hide");
//         console.log(elements);
//         console.log(elements[0]);
//         elements[0].classList.add("btn");
//         elements[0].classList.add("btn-primary");
// });
//     const isElementLoaded = async selector => {
//       while ( document.querySelector(selector) === null) {
//         await new Promise( resolve =>  requestAnimationFrame(resolve) )
//       }
//       return document.querySelector(selector);
//     };
//
//     // I'm checking for a specific class .file-item and then running code. You can also check for an id or an element.
//     isElementLoaded('.show-hide').then((selector) => {
//         // Run code here.
//         var elements = document.getElementsByClassName("show-hide");
//         console.log(elements);
//         console.log(elements[0]);
//         elements[0].classList.add("btn");
//         elements[0].classList.add("btn-primary");
//     });