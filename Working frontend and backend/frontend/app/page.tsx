
import ImageUpload from "@/components/uploadImage";

export default function Home() {
  
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="relative flex place-items-center before:absolute before:h-[300px] before:w-full before:-translate-x-1/2 before:rounded-full before:bg-gradient-radial before:from-white before:to-transparent before:blur-2xl before:content-[''] after:absolute after:-z-20 after:h-[180px] after:w-full after:translate-x-1/3 after:bg-gradient-conic after:from-sky-200 after:via-blue-200 after:blur-2xl after:content-[''] before:dark:bg-gradient-to-br before:dark:from-transparent before:dark:to-blue-700 before:dark:opacity-10 after:dark:from-sky-900 after:dark:via-[#0141ff] after:dark:opacity-40 sm:before:w-[480px] sm:after:w-[240px] before:lg:h-[360px]">
        <div className="relative z-10 flex flex-col items-center justify-between w-full max-w-5xl p-8 space-y-8 text-center rounded-xl shadow-lg backdrop-blur dark:bg-white/10">
          <label className="text-4xl font-bold text-gray-500 dark:text-gray-100">
            Image Dehazer
          </label>
          <p className="text-lg text-gray-700 dark:text-gray-300">
            Remove haze from images using deep learning.
          </p>
          <ImageUpload/>
        </div>
      </div>
    </main>
  );
}
