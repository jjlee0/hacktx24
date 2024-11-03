import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:io';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Image Classifier',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: ImageClassifierScreen(),
    );
  }
}

class ImageClassifierScreen extends StatefulWidget {
  const ImageClassifierScreen({super.key});

  @override
  _ImageClassifierScreenState createState() => _ImageClassifierScreenState();
}

class _ImageClassifierScreenState extends State<ImageClassifierScreen> {
  File? _image;
  String _result = '';
  bool _isLoading = false;

  final ImagePicker _picker = ImagePicker();

  // Function to pick an image from the gallery
  Future<void> _pickImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
    }
  }

  // Function to upload the image to the backend
  Future<void> _uploadImage() async {
    if (_image == null) return;

    setState(() {
      _isLoading = true;
      _result = '';
    });

// Define the base URL based on the platform
String getBackendUrl() {
  // Physical device - replace with your actual local IP address
  const String physicalDeviceUrl = "http://192.168.1.189:5000"; 
  return physicalDeviceUrl;

  // // Platform-specific URLs
  // if (Platform.isAndroid) {
  //   return "http://10.0.2.2:5000"; // Android emulator
  // } else if (Platform.isIOS) {
  //   return "http://127.0.0.1:5000"; // iOS simulator
  // } else {
  //   return physicalDeviceUrl; // Fallback for physical devices
  // }
}

// Set up the API URL
String apiUrl = getBackendUrl() + '/classify';
    

    try {
      var request = http.MultipartRequest('POST', Uri.parse(apiUrl));
      request.files.add(await http.MultipartFile.fromPath('file', _image!.path));

      var response = await request.send();
      if (response.statusCode == 200) {
        final responseBody = await http.Response.fromStream(response);
        final responseData = json.decode(responseBody.body);

        setState(() {
          _result = 'Class: ${responseData['predicted_class']}\n'
              'Confidence: ${responseData['confidence']}';
        });
      } else {
        setState(() {
          _result = 'Failed to classify image';
        });
      }
    } catch (e) {
      setState(() {
        print(apiUrl);
        _result = 'Error: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Image Classifier')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _image == null
                ? const Text('No image selected')
                : Image.file(_image!, height: 200),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _pickImage,
              child: const Text('Pick Image'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _uploadImage,
              child: const Text('Upload Image'),
            ),
            const SizedBox(height: 20),
            _isLoading
                ? const CircularProgressIndicator()
                : Text(
                    _result,
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontSize: 16),
                  ),
          ],
        ),
      ),
    );
  }
}
