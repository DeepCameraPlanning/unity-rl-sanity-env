using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Waypoint : MonoBehaviour
{
    public Waypoint Next;

    public MoveType Move = MoveType.Walk;

    public float AnimatorSpeed = 1f;

    public AnimationEnum AnimationToPlay = AnimationEnum.None;

    public float Range = 0.5f;

    public Transform BodyLookAt;

    //public Transform HeadLookAt;


    public enum MoveType
    {
        Walk = 0,
        Run = 1,
    }

    public enum AnimationEnum
    {
        None,
        Talk,
    }

#if UNITY_EDITOR

    protected GUIStyle m_GUIStyle = new GUIStyle();

    protected Quaternion m_RoatationUp = Quaternion.LookRotation(Vector3.up);

    protected virtual void DrawEditorIcon()
    {
        Gizmos.DrawIcon(transform.position, "Icon_Waypoint.png", true);
    }

    void OnDrawGizmos()
    {
        Vector3 myPos = transform.position;

        DrawEditorIcon();


        mfGizmos.Color = Color.red;
        mfGizmos.DrawWireCircle(myPos, Range);

        string text = "    " + transform.name + "\nAnimation:" + AnimationToPlay.ToString();
        m_GUIStyle.fontSize = 12;
        m_GUIStyle.normal.textColor = Color.green;
        UnityEditor.Handles.Label(myPos, text, m_GUIStyle);

        if(AnimationToPlay == AnimationEnum.Talk && BodyLookAt != null)
        {
            UnityEditor.Handles.color = Color.cyan;
            Vector3 vToNext = (BodyLookAt.position - myPos);
            UnityEditor.Handles.DrawLine(myPos, BodyLookAt.position);
            if (vToNext.normalized != Vector3.zero)
            {
                float distanceToNext = vToNext.magnitude;
                UnityEditor.Handles.ArrowHandleCap(0, myPos, Quaternion.LookRotation(vToNext.normalized), distanceToNext * 0.4f, Event.current.type);
            }
        }

        if (Next != null)
        {
            UnityEditor.Handles.color = Color.magenta;
            Vector3 vToNext = (Next.transform.position - myPos);
            UnityEditor.Handles.DrawLine(myPos, Next.transform.position);
            if (vToNext.normalized != Vector3.zero)
            {
                float distanceToNext = vToNext.magnitude;
                UnityEditor.Handles.ArrowHandleCap(0, myPos, Quaternion.LookRotation(vToNext.normalized), distanceToNext * 0.4f, Event.current.type);

            }

            Vector3 NextPos = Next.transform.position;
            Vector3 toNext = Next.transform.position - myPos;
            m_GUIStyle.fontSize = 10;
            m_GUIStyle.normal.textColor = Color.blue;
            //UnityEditor.Handles.Label((NextPos + myPos) / 2, "Distance: " + (int)toNext.magnitude + "\n" + "ClimbingAngle: " + Mathf.Asin(vToNext.normalized.y) * Mathf.Rad2Deg, m_GUIStyle);
            UnityEditor.Handles.Label((NextPos + myPos) / 2, Next.Move.ToString()+"\nSpeed:" + Next.AnimatorSpeed, m_GUIStyle);
        }
        //else
        //{
        //    UnityEditor.Handles.color = Color.magenta;
        //    if (transform.forward != Vector3.zero)
        //    {
        //        UnityEditor.Handles.ArrowHandleCap(0, myPos, Quaternion.LookRotation(transform.forward), 200, Event.current.type);
        //    }
        //}

    }
#endif
}
