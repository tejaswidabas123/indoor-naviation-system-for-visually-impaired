document.addEventListener('DOMContentLoaded', function () {
    const video = document.getElementById('video');
    const destinationInput = document.getElementById('destination-input');
    const captureButton = document.getElementById('capture-button');
    const framesPerSecond = 5;

    let captureInterval;
    const capturedFrames = [];


    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                video.srcObject = stream;
                video.play();
            })
            .catch(function (error) {
                console.error('Error accessing the camera: ', error);
            });
    } else {
        console.error('getUserMedia is not supported in this browser.');
    }

    function captureAndSendFrames() {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        captureInterval = setInterval(async function () {

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const imageData = canvas.toDataURL('image/jpeg');
            sendToBackend(imageData);

            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }, 1000 / framesPerSecond);
    }

    function stopCapture() {
        clearInterval(captureInterval);
    }

    async function sendToBackend(frame) {

        const backendURL = 'http://localhost:5000/getinstuctions'


        try {
            const response = await fetch(backendURL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ frame }),
            });

            if (response.ok) {
                console.log('Frames sent successfully:', frame);
            } else {
                console.error('Failed to send frames:', response.status, response.statusText);
            }
        } catch (error) {
            console.error('Error sending frames:', error);
        }
    }


    captureButton.addEventListener('click', function () {
        if (destinationInput.value.trim() === '') {
            alert('Please enter a destination.');
            return;
        }


        stopCapture();


        captureAndSendFrames();
    });
});