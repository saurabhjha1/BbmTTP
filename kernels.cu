//All kernels defined here
#include <curand_kernel.h>
#include <cuda.h>
#include <iostream>
#include "cuPrintf.cu"
#include <float.h>


#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("\nError at %s:%d\n",__FILE__,__LINE__); \
std::cout<<"\n"<<cudaGetErrorString(x); \
return EXIT_FAILURE;}} while(0)

#define n_teams 14
#define f_teams 15 //n_teams + 1
#define n_rnds 26
#define f_rnds 28 //n_rnds + 2
#define rteams 225 //f_teams * f_teams

#define bsize 420 //f_teams * f_rnds
#define gsize 30 // 2* f_teams

#define beats 2




#define maxR 1
#define maxP 1
#define maxC 2000
#define temp_curr 4000
#define alpha 0.9999

//#define temp_end 0.001

#define blocks 8
#define threads 64


/** Device Declarations **/
int dev_n_teams; //copied using ToSymbol
int dev_n_rnds; //copied using ToSymbol
float *dev_d; //copied
float *gl_best_total_dist; // copy back using FromSymbol //??? changed to pointer
float *th_distances; // contains all the distances computed locally on thread

int * dev_best_schedule; //copy back, set zero
teams* gl_dev_team; //copied
//float*  thread_curr_schedule; //memset will declare it locally



 //holds states for random number generation


/* run separetely before any processing to get random numbers */
__global__ void setup_kernel(curandState *state, int * seeds)
{
	int id=threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed , a different  sequence number no offset*/
	curand_init(seeds[id],id,0,&state[id]);
	cuPrintf("\nseed:%d",seeds[id]);
}

__device__ unsigned int dev_rand(curandState *state ,int tid)
{
// cudaPrintfDisplay(stdout, true);
 
unsigned int num=0;
curandState localState = state[tid];
num = curand(&localState);
state[tid] = localState;
if(num<0) num= num*-1;
//cuPrintf("value is :%d",num);
return num;
}

__device__ void device_initializer()
{
  // for(int b=0; b<blocks; b++)
//        gl_best_total_dist[b]=0; //??? for loop

    //gl_total_dist=0;
}


__device__ void randomize_poly(int *p,curandState *state,int tid)
{


    //randomize the contents of the array poly

    int r = 2*n_teams; //shuffle for r number of times. choice of r

    while(r--)
    {
        int s1 = dev_rand(state,tid)%n_teams;
        int s2 = dev_rand(state,tid)%n_teams;
        //cuPrintf("\n r:%d::s1:%d::s2:%d",r,s1,s2); //working//
        int temp = p[s2];
        p[s2] = p[s1];
        p[s1] = temp;

    }



}





__device__ void swap_teams(int *schedule,curandState *state, int tid)
{

    //choose 2 random numbers and swap the corresponding teams

    int t1 = dev_rand(state,tid)%(n_teams) + 1;
    int t2 = dev_rand(state,tid)%(n_teams) + 1;

    //cout<<endl<<t1<<"AND"<<t2<<endl;

    for(int i=1; i<f_rnds/2; i++)
    {
        int temp = schedule[t1*f_rnds+i];
        schedule[t1*f_rnds+i] = schedule[t2*f_rnds+i];
        schedule[t2*f_rnds+i] = temp;
    }

    //After swap, wherever t1 is there in the schedule replace by t2 and vice-versa

    for(int i=1; i<f_teams; i++)
    {
        for(int j=1; j<f_rnds/2; j++)
        {
            if(abs(schedule[i*f_rnds+j])==t1)
                schedule[i*f_rnds+j] = schedule[i*f_rnds+j] > 0 ? t2: -t2;

            else if(abs(schedule[i*f_rnds+j])==t2)
                schedule[i*f_rnds+j] = schedule[i*f_rnds+j] > 0 ? t1: -t1;

        }
    }
}

__device__ void home_away(int *schedule,curandState *state, int tid)
{

//int n1 = n_teams * n_rnds;
    //int r1 = 1; //rand()%n1 + 1;

    //make home -> away and away->home for any r1 entries
    //while(r1>0)
    {
        int r2 = dev_rand(state,tid)%(n_teams) + 1;
        int r3 = dev_rand(state,tid)%(n_rnds/2) + 1;

        int p = abs(schedule[r2*f_rnds+r3]);
        schedule[r2*f_rnds+r3] = - schedule[r2*f_rnds+r3];

        //since schedule[r2][r3] was changed that will have an effect on schedule[p][r3]
        schedule[p*f_rnds+r3] = -schedule[p*f_rnds+r3];

        //r1--;

    }
}


__device__ void swap_rounds(int *schedule,curandState *state, int tid)
{
    int r1 = dev_rand(state,tid)%(n_rnds/2) + 1;
    int r2 = dev_rand(state,tid)%(n_rnds/2) + 1;



    for(int i=1; i<f_teams; i++)
    {
        int temp = schedule[i*f_rnds+r1];
        schedule[i*f_rnds+r1] = schedule[i*f_rnds+r2];
        schedule[i*f_rnds+r2] = temp;
    }


}


/*XXXX change made till end of row swap
__device__ bool find(int *a, int lsize, int i)
{
    for(int j=0; j<lsize; j++)
        if(a[j] == i)
            return true;
    return false;
}
*/

__device__ bool find(int *a, int lsize, int i)
{
    for(int j=0; j<lsize; j++)
        if(a[j] == i)
            return true;
    return false;
}


__device__ int column_correction(int *schedule, int *seen_row,int *seen_col,int c1, int n1, int n2, int size)
{
    int r=-1;

    for(int i=1; i<f_teams; i++)
    {
        bool isPresent = find(seen_row, size, i);

        if(isPresent == false)
        {
            if(abs(schedule[i*f_rnds+c1]) == abs(n2))
            {
                r = i;
                schedule[i*f_rnds+c1] =  n1;
            }
        }
    }



    for(int i=1; i<f_teams; i++)
    {
        if(schedule[i*f_rnds+c1] == n1)
            schedule[abs(n1)*f_rnds+c1] = n1 > 0 ? -i : i ;

        if(schedule[i*f_rnds+c1] == n2)
            schedule[abs(n2)*f_rnds+c1] = n2 > 0 ? -i : i ;

    }

    return r;
}


__device__ int row_correction(int *schedule, int *seen_row,int *seen_col, int r1, int n1, int n2, int size)
{
    int c=-1;


    for(int i=1; i<= (n_rnds/2); i++)
    {
        bool isPresent = find(seen_col, size, i);

        if(isPresent == false)
        {
            if(abs(schedule[r1*f_rnds+i]) == abs(n1))
            {
                c = i;
                schedule[r1*f_rnds+i] =  n2;
            }
        }
    }

    return c;
}



__device__ void row_swap(int *schedule,curandState* state, int tid)
{
    //swap two elements in a row

    int seen_row[100];
    int seen_col[100];
    int r=0, c=0; //counters for seen_row and seen_col


    int t1 = dev_rand(state,tid)%n_teams + 1;

    int c1 = dev_rand(state,tid)%(n_rnds/2) + 1;
    int c2 = dev_rand(state,tid)%(n_rnds/2) + 1;

    //t1=1; c1=1;

    if(c1==c2)
    {
        if(c1==n_rnds/2)
            c2 = c1 - 1;
        else
            c2 = c1 + 1;
    }


    int n1 = schedule[t1*f_rnds+c1];
    int n2 = schedule[t1*f_rnds+c2];


    schedule[t1*f_rnds+c1] = n2;
    schedule[t1*f_rnds+c2] = n1;


    seen_row[r++] = t1;


    int r1=0, r2=0, k,l;



    do
    {
        r1= column_correction(schedule,seen_row,seen_col, c1,  n1, n2, r);
        seen_col[c++] = c1;

        r2= column_correction(schedule,seen_row,seen_col, c2,  n2,  n1,  r);
        seen_col[c++] = c2;


        if(r1==r2)
            seen_row[r++] = r1;


        if(r1!=r2)
        {
            c1 = row_correction(schedule,seen_row,seen_col, r1,  n1, n2, c);
            seen_row[r++] = r1;


            c2 = row_correction(schedule,seen_row,seen_col, r2,  n2,  n1, c);
            seen_row[r++] = r2;
        }

    }
    while(r1!=r2);

}











__device__ void select_random_schedule(int *sch,curandState *state, int tid)
{
    int p = dev_rand(state,tid)%4+ 1; // 4 operations to randomize a schedule

    switch(p) //change this
    {
    case 1:
        swap_teams(sch,state,tid);
        break;
    case 2:
        home_away(sch,state,tid);
        break;
    case 3:
        row_swap(sch,state,tid);
        break;
    case 4:
        swap_rounds(sch,state,tid);
        break;

    default:
	swap_rounds(sch,state,tid);
    }

}


__device__ float dev_dist_travelled(int *sch, dev_teams *l_dev_team, float* dev_d) //gives the total distance travelled in each schedule
{

    float dist=0.0;

    for(int i=1; i<f_teams ; i++)
    {
        //consider the ith row in the schedule matrix.
        l_dev_team[i].reset(i);

        for(int j=1; j<f_rnds/2; j++) // does the first half
        {
            int p = l_dev_team[i].present_pos;
            int q = sch[i*f_rnds+j] ;// -- gives the destination where the game will be played

            if(q<0)
            {
                //cout<<"\ninside!!!";
                l_dev_team[i].distance_covered = l_dev_team[i].distance_covered + dev_d[p*f_teams+abs(q)]; //?? check this implementation
                l_dev_team[i].present_pos = abs(q);
            }

            else
            {
                int h = l_dev_team[i].home;
                l_dev_team[i].distance_covered = l_dev_team[i].distance_covered + dev_d[p*f_teams+h];
                l_dev_team[i].present_pos = h;
            }
        }


        for(int j=1; j<f_rnds/2; j++) // does the second half
        {
            int p = l_dev_team[i].present_pos;
            int q = sch[i*f_rnds+j] ;// -- gives the destination where the game will be played


            if(q>0)
            {
                l_dev_team[i].distance_covered = l_dev_team[i].distance_covered + dev_d[p*f_teams+q];
                l_dev_team[i].present_pos = q;
            }

            else
            {
                int h = l_dev_team[i].home;
                l_dev_team[i].distance_covered = l_dev_team[i].distance_covered + dev_d[p*f_teams+h];
                l_dev_team[i].present_pos = h;
            }

        }
        //everybody traveles back home after the entire schedule
        int p = l_dev_team[i].present_pos;
        int h = l_dev_team[i].home;

        l_dev_team[i].distance_covered = l_dev_team[i].distance_covered + dev_d[p*f_teams+h];

        dist = dist+l_dev_team[i].distance_covered;

    }


    return dist;

}

__device__ int min_index(float *numbers,int size) //EE
{
    int b = blockIdx.x;
    //cuPrintf("blocks:%d",b);
    float dmin=FLT_MAX;
    int index=-1;
    for(int i=b*threads; i<(b*threads) + size; i++)
    {
        if(numbers[i]<=dmin)
        {
            dmin=numbers[i];
            index=i;

        }
    }
    return index;
}


__device__ int check_no_of_home_away(int *schedule,int na=n_rnds/2)
{
    //int ch_pos=0; //to see how many cont home matches in the beginining
    // int ch_neg=0;

    //int flag_pos =0; //
    //int flag_neg =0;

    for(int i=1; i<f_teams; i++)
    {
        for(int j=1; j<=(n_rnds/2); j++)
        {
            int ch_pos=0;
            int ch_neg=0;

            for(int k=j; k<j+na+1; )
            {
                if(k<j+na+1)
                {
                    if(k>n_rnds/2)
                    {
                        int p = k%(n_rnds/2);

                        if(schedule[i*f_rnds+p] < 0)
                            ch_pos++;

                        if(schedule[i*f_rnds+p] > 0)
                            ch_neg++;

                    }


                    else
                    {
                        if(schedule[i*f_rnds+k] > 0)
                            ch_pos++;

                        if(schedule[i*f_rnds+k] < 0)
                            ch_neg++;
                    }

                }

                k++;
            }


            if(ch_pos > na || ch_neg > na) //implies that the number of consecutive home/away > na
                return -1;
        }
    }


    return 1;
}

__device__ int chk_schedule(int*schedule,int na=n_rnds/2)
{
    int nha = check_no_of_home_away(schedule,na);

    if(nha == -1 )
        return -1;  //implies the schedule is not  feasible
    else
        return 1;  //implies the schedule is feasible
    //return 0;
}




__global__ void annealing(curandState * state, teams* gl_dev_team,float* dev_d, int * dev_best_schedule,float * th_distances,float* gl_best_total_dist,int c=1)
{

   // int counter=1;
    int mIndex;

    int b = blockIdx.x; //???

    int id=threadIdx.x+blockIdx.x*blockDim.x;
    float local_best_distance=0;
    float local_curr_distance=0;
    float total_dist;
    device_initializer();
    int tid= threadIdx.x + blockIdx.x*blockDim.x;
    dev_teams l_dev_team[f_teams];

    for(int i=1; i<f_teams; i++)
    {
        l_dev_team[i].present_pos=gl_dev_team[i].present_pos;
        l_dev_team[i].distance_covered=gl_dev_team[i].distance_covered;
        l_dev_team[i].home=gl_dev_team[i].home;

        //working//
    }


    //XXXX __syncthreads();

    /*generate initial schedule*/

    int thread_curr_schedule[f_teams*f_rnds];
    int thread_schedule[f_teams*f_rnds];
    int thread_best_schedule[f_teams*f_rnds];
    for(int i=1; i<f_teams; i++)
    {
        for(int j=1; j<f_rnds; j++)
        {
            thread_schedule[i*f_rnds+j]=0;
            thread_best_schedule[i*f_rnds+j]=0;
            thread_curr_schedule[i*f_rnds+j]=0;
        }
    }
    // cuPrintf("\n%d",thread_schedule[0]);
    int t1, t2;

    int poly[n_teams];
    int dummy_poly[n_teams];

    for(int i=0; i<n_teams; i++)
        poly[i] = i+1;

    randomize_poly(poly,state,tid);

    for(int i=0; i<n_teams; i++)
    {
        
        dummy_poly[i] = poly[i];
        //cuPrintf("\n%d",dummy_poly[i]); //working//
        
    }
    
    
    //cudaMemcpy(dummy_poly,poly, sizeof(int)*f_teams,cudaMemcpyDeviceToDevice); can't be used inside kernels

   
    for(int i=1; i<f_rnds/2; i++)
    {
        for(int j=0; j<f_teams/2; j++)
        {
            int type = dev_rand(state,tid)%2;  // 0 - denotes home match between i and i+1 at i, 1- denotes away match
            //cuPrintf("\n%d",type); //working//
            t1 = poly[j]; //?? //vijay
            t2 = poly[ n_teams -j -1]; //?? //vijay
            //cuPrintf("\nt1:%d,t2:%d",t1,t2); //working//
            //	cuPrintf("\n%d",thread_schedule[8]);

            if(type == 0)
            {
                thread_schedule[t1*f_rnds+i] = t2;
                thread_schedule[t2*f_rnds+i] = -t1;
            }

            else
            {
                thread_schedule[t1*f_rnds+i] = -t2;
                thread_schedule[t2*f_rnds+i] = t1;

            }
            //cuPrintf("\n%d",thread_schedule[21]);

        }


        for(int k=2; k<n_teams; k++)
            poly[k] = dummy_poly[k-1];

        poly[1] = dummy_poly[n_teams - 1];



        for(int i=0; i<n_teams; i++)
            dummy_poly[i] = poly[i];

    }

    for(int i=1; i<f_teams; i++)
    {

        for(int j=1; j<f_rnds/2; j++)
        {

            thread_best_schedule[i*f_rnds+j] = thread_schedule[i*f_rnds+j];
            thread_curr_schedule[i*f_rnds+j] = thread_schedule[i*f_rnds+j];

        }
    }

    /* initial schedule generated */

    float thread_curr_total_dist;// = gl_best_total_dist; //copy global best //>>>>>>>
    float thread_best_total_dist;// = gl_best_total_dist; // initially the global best will be the thread best, after processing this may change

    float thread_total_dist=dev_dist_travelled(thread_schedule,l_dev_team,dev_d);
    //th_distances[id]=dev_dist_travelled(thread_schedule,l_dev_team,dev_d);

    if( chk_schedule(thread_schedule,3) == -1 )
        thread_total_dist = 10000000.00; //implies that the genrated randon schedule is not a feasible one


    thread_curr_total_dist = thread_total_dist;
    thread_best_total_dist = thread_total_dist;

    float temp = temp_curr;
    float delta;
    float rnum;
    int chk; 

	
	int reheat =1;
	int counter =0;
	int phase=0;	
	
	float bestTemp = temp_curr;
    
    while(reheat<=maxR)
	{	
		      
	
        	if(reheat%beats==0)
        	{
          	  __syncthreads();
           	 th_distances[id]=thread_best_total_dist;

            	mIndex=min_index(th_distances,threads);
            	gl_best_total_dist[b]=th_distances[mIndex]; //???
            	if(tid==mIndex)
            	{
               		 for(int i=1; i<f_teams; i++)
                    		for(int j=1; j<f_rnds/2; j++)
                    		{
                        		dev_best_schedule[b*bsize + i*f_rnds + j] = thread_best_schedule[i*f_rnds+j]; //???
			

                    		}
            	}
            	__syncthreads();
	
            	for(int i=1; i<f_teams; i++)
                	for(int j=1; j<f_rnds/2; j++)
                	{
                   	 thread_curr_schedule[i*f_rnds+j] = dev_best_schedule[b*bsize + i*f_rnds+j];//???
                   	 thread_best_schedule[i*f_rnds+j] = dev_best_schedule[b*bsize + i*f_rnds+j]; //???
                    	 thread_schedule[i*f_rnds+j] = dev_best_schedule[b*bsize + i*f_rnds+j]; //???
               		 }
        	}
      
		
		
		phase = 0;
	
		while(phase < maxP)
		{
			counter = 0;
			
			while(counter < maxC)
			{
				for(int i=1; i<f_teams; i++)
            				for(int j=1; j<f_rnds/2; j++)
            				{
                				thread_schedule[i*f_rnds+j]=thread_curr_schedule[i*f_rnds+j];
                			
                			}
                			
                		
                		select_random_schedule(thread_schedule,state,tid);
                		
                		thread_total_dist = dev_dist_travelled(thread_schedule,l_dev_team,dev_d);
            			thread_curr_total_dist = dev_dist_travelled(thread_curr_schedule,l_dev_team,dev_d);
            			
            			delta = thread_total_dist - thread_curr_total_dist;
            			
            			rnum = float(dev_rand(state,tid)%100000)/100000.f;
            			
            			if(delta < 0 || exp(-delta/temp) > 0.87)
           			 {

                    			chk = chk_schedule(thread_schedule,3);
                    			
                    			 if(chk==1)
                    			{
                				thread_curr_total_dist = thread_total_dist;
                
               					 for(int i=1; i<f_teams; i++)
                    					for(int j=1; j<f_rnds/2; j++)
                       						 thread_curr_schedule[i*f_rnds+j] = thread_schedule[i*f_rnds+j];
                       						 
                       				
                       				thread_curr_total_dist = thread_total_dist;
                       				
                       				if(thread_curr_total_dist < thread_best_total_dist)  
                       				{
                    
                        				//we got a better solution make it thread_best_total_dist
                       
                        				thread_best_total_dist=thread_curr_total_dist;
                       					
                       					 for(int i=1; i<f_teams; i++)
                            					for(int j=1; j<f_rnds/2; j++)
                            		                             thread_best_schedule[i*f_rnds+j] = thread_curr_schedule[i*f_rnds+j]; 
                            		                             
                            		         }
                            		         
                            		         bestTemp = temp;
                            		         
                            		  }
                            		  
                            		//  else
                            		  //counter++;
                            		  	
                            		  	
                            		  	
                            	}
                            	
                            	counter++;


			}//end while counter
			
			temp = temp * alpha;
			phase++;
			
		}//end while
		
		reheat++;
       	temp = 2*bestTemp;	                            	
     
   	}//end while
   	
    //synchronize the best possible solution to all threads at this juncture, if that best schedule was not on the beats
    
    th_distances[id]=thread_best_total_dist;
    __syncthreads();
    mIndex=min_index(th_distances,threads);
    gl_best_total_dist[b]=th_distances[mIndex]; //???

    if(tid==mIndex)
    {
        for(int i=1; i<f_teams; i++)
            for(int j=1; j<f_rnds/2; j++)
            {
                dev_best_schedule[b*f_teams*f_rnds + i*f_rnds+j] = thread_best_schedule[i*f_rnds+j]; //???

            }
    }


}

