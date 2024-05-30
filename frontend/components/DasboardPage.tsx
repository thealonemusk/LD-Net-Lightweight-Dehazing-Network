"use client";
import { useState } from "react";


// Headless UI 2.x for React, for more info and examples you can check out https://github.com/tailwindlabs/headlessui

import Image from "next/image";

export default function DashboardPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [url, setUrl] = useState('');
  const handleImageUpload = (event:any) => {
    setSelectedImage(event.target.files[0]);
  };

  const handleImageSend = async () => {
    const formData = new FormData();
    if(selectedImage==""){
        console.log("No image selected")
        return;
    }else{
        formData.append('image', selectedImage??"");
    }

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/image-upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setUrl(url);
        console.log(`URL------------><><><><><><><----------------${url}`)
      } else {
        console.error('Image upload failed');
      }
    } catch (error) {
      console.error('Image upload failed:', error);
    }
  }

  return (
    <>
    <section className="max-w-lg mx-auto my-8">
        <div className="flex items-center justify-center w-full mb-4">
            <label
                htmlFor="image-upload"
                className="flex flex-col items-center justify-center w-full p-10 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer hover:bg-gray-800 hover:bg-opacity-40 transition duration-300 ease-in-out"
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
                    <p className="text-xs text-gray-500 dark:text-gray-400">SVG, PNG, JPG, or GIF</p>
                </div>
                <input
                    id="image-upload"
                    name="image-upload"
                    type="file"
                    className="hidden"
                    onChange={handleImageUpload}
                />
            </label>
        </div>
        <button
            onClick={handleImageSend}
            className="w-full p-3 text-white bg-gray-800 rounded-lg hover:bg-gray-700 transition duration-300 ease-in-out"
        >
            Upload
        </button>
        {(
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="flex items-center justify-center mt-4">
                    <Image
                        width={300}
                        height={300}
                        src={selectedImage?URL.createObjectURL(selectedImage):""}
                        alt="Selected Image"
                        className="object-center object-contain rounded-lg shadow-md max-w-full max-h-64"
                    />
                    
                </div>
                <div className="flex items-center justify-center mt-4">
                    <Image
                        width={300}
                        height={300}
                        src={url}
                        alt="Dehazed Image"
                        className="object-center object-contain rounded-lg shadow-md max-w-full max-h-64"
                    />
                </div>
            </div>
        )}
    </section>
    </>
  );
}
