# American Sign Language Recognition System

The American Sign Language Recognition System is a project aimed at recognizing and translating American Sign Language (ASL) gestures into text. This system leverages computer vision and machine learning techniques to interpret ASL gestures, providing a valuable tool for communication with the deaf and hard-of-hearing community.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
The American Sign Language Recognition System is designed to translate ASL gestures into text using deep learning models. It aims to bridge the communication gap between ASL users and non-ASL users.

## Features
- Real-time ASL gesture recognition
- High accuracy with deep learning models
- User-friendly interface
- Docker support for easy deployment

## Installation

### Prerequisites
- Python 3.x
- Docker (optional, for containerized deployment)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/aayush1693/American-Sign-Language-Recognition-System.git
   cd American-Sign-Language-Recognition-System
   ```
2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Setup (Optional)
To run the project using Docker:
1. Build the Docker image:
   ```bash
   docker build -t asl-recognition-system .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 asl-recognition-system
   ```

## Usage
1. Ensure your webcam is connected.
2. Run the ASL recognition script:
   ```bash
   python run.py
   ```
3. The system will start recognizing ASL gestures and translating them into text displayed on the screen.

## Screenshots
(Add screenshots here to show the application in action)

## Contributing
We welcome contributions to enhance the American Sign Language Recognition System. To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Description of your changes"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Create a pull request, describing your changes in detail.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## Acknowledgements
The project uses various open-source libraries and tools. Special thanks to the contributors and the ASL community for their support.

