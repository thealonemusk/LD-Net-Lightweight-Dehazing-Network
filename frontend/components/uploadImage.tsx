"use client"
import React, { useState } from 'react';

const ImageUpload = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const handleImageUpload = (event:any) => {
    setSelectedImage(event.target.files[0]);
  };

  const handleImageSend = async () => {
    const formData = new FormData();
    formData.append('image', selectedImage??'');

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/image-upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        console.log(`URL------------><><><><><><><----------------${url}`)
      } else {
        console.error('Image upload failed');
      }
    } catch (error) {
      console.error('Image upload failed:', error);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-center w-full mb-4">
        <label
          htmlFor="image-upload"
          className="flex flex-col items-center justify-center w-full p-10 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer  hover:bg-gray-800 hover:bg-opacity-40"
        >
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <svg
              aria-hidden="true"
              className="w-10 h-10 mb-3 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
            <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
              <span className="font-semibold">Click to upload</span> or drag and drop
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">SVG, PNG, JPG or GIF</p>
          </div>
          <input
            id="image-upload"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleImageUpload}
          />
        </label>
      </div>

      {selectedImage && (
        <>
        <div className="flex items-center justify-center">
          <img
            src={URL.createObjectURL(selectedImage)}
            alt="Selected Image"
            className="object-center object-contain rounded-lg shadow-md" />
        </div>
        <button className='flex-1 p-2 bg-blue-800 rounded-xl m-5 max-w-50' onClick={handleImageSend}>
          Upload
        </button>
        </>
      )}
    </div>
  );
};

export default ImageUpload;