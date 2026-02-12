using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;





public class PointToPointAgent : Agent
{

    // components
    private Rigidbody rb;
    public Transform startPoint;
    public Transform target;
    public float moveSpeed = 1f;
    //private float episodeTimer;
    //public float maxEpisodeTime = 10f; // seconds
    //moving to steps timeout instead of time
    private int stepCount;
    private float prevDistance;
    public int maxEpisodeSteps = 1000; // steps

    //The travel speed to try and achieve
    private float m_TargetTravelSpeed = m_maxTravelSpeed;

    const float m_maxTravelSpeed = 15; //The max travel speed

    //The current target travel speed. Clamped because a value of zero will cause NaNs
    public float TargetTravelSpeed
    {
        get { return m_TargetTravelSpeed; }
        set { m_TargetTravelSpeed = Mathf.Clamp(value, .1f, m_maxTravelSpeed); }
    }


    // initialization
    
    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {   // reset
        stepCount = 0;
        prevDistance = Vector3.Distance(transform.position, target.position);
        StartCoroutine(SnapToStartNextFrame());
        TargetTravelSpeed = Random.Range(0.5f, m_maxTravelSpeed);

        
    }

    private System.Collections.IEnumerator SnapToStartNextFrame()
    {
        // wait one frame so any other scripts initialize first
        yield return null;
        // reset position and velocity
        transform.position = startPoint.position;
        // reset rotation
        transform.rotation = startPoint.rotation;
        // reset velocity
        rb.velocity = Vector3.zero;
        // reset angular velocity
        rb.angularVelocity = Vector3.zero;

    }


    // observations
    public override void CollectObservations(VectorSensor sensor)
    {
        //new reward sensor creation
        
        // convert to local frame
        //Vector3 localVel = transform.InverseTransformDirection(vel);
        // get velocity to target in local frame
        Vector3 toTarget = (target.position - transform.position).normalized;
        // create a rotation that looks at the target
        Quaternion targetFrame = Quaternion.LookRotation(toTarget, Vector3.up);
        // convert velocity to target frame
        Vector3 localVelToTarget = Quaternion.Inverse(targetFrame) * rb.velocity;
        // desired velocity vector in world frame towards target
        Vector3 velGoal = toTarget * TargetTravelSpeed;
        // average velocity
        Vector3 avgVel = rb.velocity;
        // target position in local frame
        Vector3 localTargetPos = transform.InverseTransformPoint(target.position);

        //old observations commented out

/*         //// adding observations
        /// position of agent 
        sensor.AddObservation(transform.position);
        // position of target 
        sensor.AddObservation(target.position);
        // velocity of agent 
        sensor.AddObservation(localVel);
 */

        // velocity of agent relative to target 
        sensor.AddObservation(localVelToTarget);
        // desired velocity
        sensor.AddObservation(Vector3.Distance(velGoal, avgVel) / m_maxTravelSpeed);
        // target position in local frame
        sensor.AddObservation(localTargetPos);

        // agent velocity in local frame
       // sensor.AddObservation(transform.InverseTransformDirection(rb.velocity));

        // direction to target
        sensor.AddObservation(transform.InverseTransformDirection(toTarget)); // direction-to-target in local frame


    }

    // actions and rewards

    public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
    {
        //distance between our actual velocity and goal velocity
        var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, TargetTravelSpeed);

        //return the value on a declining sigmoid shaped curve that decays from 1 to 0
        //This reward will approach 1 if it matches perfectly and approach zero as it deviates
        return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / m_maxTravelSpeed, 2), 2);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // increment step count
        stepCount++;
        
        // check for max steps
        if (stepCount >= maxEpisodeSteps)
        {
            EndEpisode();
            return;
        }

        //move and turn values
        float move = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        // scale turn down to small values for smoother turning
        float turn = Mathf.Clamp(actions.ContinuousActions[1], -.01f, .01f);

        
      

        // apply movement
        rb.AddForce(transform.forward * move * moveSpeed, ForceMode.Acceleration);

        // calculate yaw rotation
        float yaw = turn * 180f * Time.fixedDeltaTime;
        // apply rotation
        rb.MoveRotation(rb.rotation * Quaternion.Euler(0f, yaw, 0f));

        //old reward shaping code commented out


/*         // creating a bunch of variables to potentially use for reward shaping 

        // direction to target
        Vector3 toTarget = (target.position - transform.position).normalized;
        float facing = Vector3.Dot(transform.forward, toTarget); // -1..1
      
        // reward calculation
        // progress towards target since last step
        
        // clamp progress to -1..1
        float p = Mathf.Clamp(progress, -1f, 1f);
        //weights
        float wprogress = 0.7f;
        float wfacing = .01f;
        // reward instantiation into single value
        float r = p * wprogress; 
        r += Mathf.Max(0f, facing) * wfacing;
        r += -0.001f * Mathf.Abs(turn); // small penalty for turning to encourage efficiency 
        Debug.Log($"progress {progress}, facing {facing}, reward {r}, step {stepCount}");
        AddReward(r);             
     
         */


        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        float progress = prevDistance - distanceToTarget;
        
        //new reward shaping sigmoid line curve code
        Vector3 toTarget = (target.position - transform.position).normalized;
        Vector3 velGoal = toTarget * TargetTravelSpeed;
        
        float r = GetMatchingVelocityReward(velGoal, rb.velocity);
    
        
        AddReward(r*.01f); // scaled down for stepwise addition
        AddReward(progress); // reward for progress towards target
        //Debug.Log($"progress {progress}, velocity reward {r}, total reward this step {r*.01f + progress}");
        prevDistance = distanceToTarget;
        // Success condition
        if (distanceToTarget < 1.0f)
        {
            float proxReward = 20.0f;
            AddReward(proxReward);
            Debug.Log($"Reached target in {stepCount} steps and {distanceToTarget}.");
            EndEpisode();
        }
    }
}
