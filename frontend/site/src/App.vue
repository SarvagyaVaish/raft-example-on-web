<template>
  <section class="bg-white dark:bg-gray-900">
    <div class="py-8 px-4 mx-auto max-w-2xl lg:py-16">
      <h1 class="mb-4 text-3xl font-bold text-gray-900 dark:text-white">Optical Flow Estimation using RAFT</h1>
      <p class="mb-12 text-lg text-gray-900 dark:text-white">Upload two images to estimate the optifal flow using RAFT, a
        deep learning model.</p>

      <form @submit.prevent="runRAFT" enctype="multipart/form-data">
        <div class="mb-12 grid gap-4 sm:grid-cols-2 sm:gap-6">
          <div class="w-full">
            <fwb-file-input accept="image/*" v-model="frame1ImageModel" label="Frame 1" @change="handleFrame1Change" />
          </div>
          <div class="w-full">
            <fwb-file-input accept="image/*" v-model="frame2ImageModel" label="Frame 2" @change="handleFrame2Change" />
          </div>

          <div class="w-full">
            <img v-if="frame1Image" :src="frame1Image" />
          </div>
          <div class="w-full">
            <img v-if="frame2Image" :src="frame2Image" />
          </div>
        </div>

        <div class="mb-12 flex justify-center">
          <fwb-button type="submit" color="green">Run RAFT</fwb-button>
        </div>

        <div class="mb-12 flex justify-center">
          <div class="w-full">
            <img v-if="raftResultImage" :src="raftResultImage" />
          </div>
        </div>

      </form>
    </div>
  </section>
</template>

<script>
import { ref } from 'vue'
import { FwbFileInput } from 'flowbite-vue'
import { FwbButton } from 'flowbite-vue'

export default {
  components: {
    FwbFileInput,
    FwbButton
  },

  data() {
    return {
      frame1Image: null,
      frame2Image: null,
      frame1ImageModel: null,
      frame2ImageModel: null,
      raftResultImage: null,
    }
  },

  methods: {
    runRAFT() {
      // Check if both frames are selected
      if (this.frame1Image && this.frame2Image) {
        // Create FormData object to append files
        const formData = new FormData()
        formData.append('first_image', this.frame1ImageModel)
        formData.append('second_image', this.frame2ImageModel)

        // Make an API call using fetch or your preferred HTTP library
        fetch('https://raft-web-backend-wnch5tzs6q-uc.a.run.app/process', {
          method: 'POST',
          body: formData,
          // You may need to set headers based on your API requirements
        })
          .then(response => response.blob()) // Assuming the response is a blob
          .then(blob => {
            // Create a URL for the blob
            const imageURL = URL.createObjectURL(blob);
            this.raftResultImage = imageURL;

            // Handle the API response
            console.log('API Response:', imageURL);
          })
          .catch(error => {
            // Handle errors
            console.error('API Error:', error);
          });
      } else {
        // Handle the case where one or both frames are not selected
        console.warn('Please select both frames before running RAFT.');
      }
    },

    handleFrame1Change(event) {
      const file = event.target.files[0];
      this.frame1Image = URL.createObjectURL(file);
    },

    handleFrame2Change(event) {
      const file = event.target.files[0];
      this.frame2Image = URL.createObjectURL(file);
    },
  },

  mounted() {
    console.log('Mounted')
  },
}

</script>
