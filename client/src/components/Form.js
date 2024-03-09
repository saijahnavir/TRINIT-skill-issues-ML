import {React, useState} from 'react'

function Form() {

    const [textarea, setTextarea] = useState("");
    
      const handleChange = (event) => {
        setTextarea(event.target.value)
      }

  return (
    <div className='text-black-800 text-xl inline'>
        <h1 className='text-center font-bold text-3xl mt-10'>Sexual Harassment Input Form </h1>
        <form className='bg-indigo-50 m-10 p-10 flex flex-col rounded'>
            <label className='mb-10 text-center font-light'>Enter your experience : </label>
            <textarea value={textarea} placeholder='Enter description here' onChange={handleChange} className='text-black text-base rounded ring-4 ring-indigo-200 ring-offset-0 focus:outline-none w-90'  />
            <button className=' place-self-center mt-10 bg-black w-24 p-2 rounded text-white hover:border hover:border-black hover:bg-indigo-50 hover:text-black'>Submit</button>
        </form>
        <p className='text-center underline'>Prediction :</p>
    </div>
  )
}

export default Form